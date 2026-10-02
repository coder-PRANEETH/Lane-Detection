import unittest

import numpy as np

from road.lane import LaneEstimator


def road_mask(shift=0.0):
    """A perspective road whose lateral position is shift ground units."""
    h, w, horizon = 180, 320, 60
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(70, h):
        depth = y - horizon
        centre = (w - 1) / 2 + shift * depth
        left = max(0, int(centre - 0.6 * depth))
        right = min(w, int(centre + 0.6 * depth) + 1)
        mask[y, left:right] = 1
    return mask


class LaneSmoothingTests(unittest.TestCase):
    def estimator(self, **kwargs):
        return LaneEstimator(horizon=1 / 3, **kwargs)

    def assert_rate_bound(self, previous, result, dt):
        self.assertLessEqual(abs(result.offset - previous[0]), 0.35 * min(dt, 0.1) + 1e-12)
        self.assertLessEqual(abs(result.steer_deg - previous[1]), 20 * min(dt, 0.1) + 1e-12)

    def test_startup_step_and_persistent_change(self):
        lane = self.estimator()
        previous = (0.0, 0.0)
        for i in range(180):
            result = lane(road_mask(0.6 if i < 60 else -0.6))
            self.assert_rate_bound(previous, result, 1 / 30)
            previous = result.offset, result.steer_deg
        self.assertTrue(result.found)
        self.assertFalse(result.held)
        self.assertLess(result.offset, -0.45)
        self.assertAlmostEqual(result.offset, result.raw_offset, delta=0.005)
        self.assertAlmostEqual(result.steer_deg, result.raw_steer_deg, delta=0.02)

    def test_single_spike_freezes_the_trusted_lane(self):
        lane = self.estimator()
        for _ in range(10):
            before = lane(road_mask())
        spike = lane(road_mask(0.65))
        self.assertTrue(spike.held)
        self.assertEqual(before.offset, spike.offset)
        self.assertEqual(before.steer_deg, spike.steer_deg)
        np.testing.assert_array_equal(before.path, spike.path)
        after = lane(road_mask())
        self.assertFalse(after.held)
        self.assertLess(abs(after.offset), 0.01)

    def test_dropout_freezes_and_reacquisition_has_no_jump(self):
        lane = self.estimator(hold_frames=3)
        for _ in range(8):
            before = lane(road_mask(0.6))
        missing = np.zeros_like(road_mask())
        for i in range(10):
            result = lane(missing)
            self.assertEqual(result.found, i < 3)
            self.assertEqual(result.held, i < 3)
            self.assertEqual(result.offset, before.offset)
            self.assertEqual(result.steer_deg, before.steer_deg)
            if result.held:
                np.testing.assert_array_equal(result.path, before.path)
                self.assertEqual(result.target, before.target)
        after = lane(road_mask(-0.6), dt=5.0)
        self.assertTrue(after.found)
        self.assertFalse(after.held)
        self.assert_rate_bound((before.offset, before.steer_deg), after, 5.0)

    def test_variable_frame_intervals(self):
        lane = self.estimator()
        previous = (0.0, 0.0)
        for dt in [0.01, 0.04, 0.1, 0.002, 4.0] * 5:
            result = lane(road_mask(0.6), dt=dt)
            self.assert_rate_bound(previous, result, dt)
            previous = result.offset, result.steer_deg

    def test_inconsistent_rejections_never_confirm_each_other(self):
        lane = self.estimator()
        original = np.array([0.0, 0.0, 1.0, 80.0])
        self.assertTrue(lane._remember(original))
        for i in range(30):
            candidate = np.array([0.0, 0.8 if i % 2 else -0.8, 1.0, 80.0])
            self.assertFalse(lane._remember(candidate))
        np.testing.assert_array_equal(lane.recent[0], original)
        for i in range(5):
            accepted = lane._remember(np.array([0.0, 0.8, 1.0, 80.0]))
        self.assertTrue(accepted)
        self.assertAlmostEqual(np.median(lane.recent, axis=0)[1], 0.8)

    def test_missing_frame_breaks_change_confirmation(self):
        lane = self.estimator()
        lane(road_mask())
        for _ in range(4):
            self.assertTrue(lane(road_mask(0.6)).held)
        lane(np.zeros_like(road_mask()))
        self.assertEqual(len(lane.rejected), 0)
        self.assertTrue(lane(road_mask(0.6)).held)

    def test_mask_range_and_nonfinite_pixels(self):
        mask = road_mask()
        a, b = self.estimator()(mask), self.estimator()(mask * 255)
        self.assertEqual(a.offset, b.offset)
        self.assertEqual(a.steer_deg, b.steer_deg)
        self.assertEqual(a.obstacle, b.obstacle)
        result = self.estimator()(np.full(mask.shape, np.nan))
        self.assertFalse(result.found)
        self.assertTrue(np.isfinite(result.offset))

    def test_estimators_are_repeatable_when_interleaved(self):
        one, two = self.estimator(), self.estimator()
        for shift in [0, 0.02, 0.4, -0.1, 0.3] * 2:
            a = one(road_mask(shift))
            self.estimator()(road_mask(-shift))
            b = two(road_mask(shift))
            self.assertEqual(a.offset, b.offset)
            self.assertEqual(a.steer_deg, b.steer_deg)
            self.assertEqual(a.held, b.held)

    def test_invalid_timing_and_configuration(self):
        lane = self.estimator()
        for dt in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                lane(road_mask(), dt=dt)
        for kwargs in [dict(fps=0), dict(average=0), dict(row_step=0), dict(hfov=180),
                       dict(max_offset_rate=-1), dict(max_dt=0)]:
            with self.assertRaises(ValueError):
                self.estimator(**kwargs)


if __name__ == "__main__":
    unittest.main()
