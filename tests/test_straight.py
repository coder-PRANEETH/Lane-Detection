"""Behavioral checks for keeping a clear, straight course away from lane centre."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from road.control import SensorSample, SteeringController, validate_control_config
from road.lane import LaneEstimator, WARMUP


def straight_lane(**changes):
    values = dict(found=True, held=False, offset=0.25, steer_deg=10.0,
                  heading_deg=0.0, straight_clearance=0.25, obstacle=None, way="")
    values.update(changes)
    return SimpleNamespace(**values)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Sensor:
    def __init__(self, clock, angle=0.0):
        self.clock, self.angle = clock, angle
        self.age = 0.0
        self.connected = True
        self.written = []

    def read(self, dt):
        return SensorSample(self.angle, self.clock() - self.age, self.connected)

    def write(self, result, timeout_s):
        self.written.append(result)

    def close(self):
        pass


class KeepStraightTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sensor = Sensor(self.clock)

    def controller(self, **config):
        return SteeringController(config, adapter=self.sensor, clock=self.clock)

    def step(self, controller, current_lane=None, dt=0.05, **kwargs):
        self.clock.now += dt
        return controller.update(current_lane or straight_lane(), dt, **kwargs)

    def test_offcentre_on_either_side_does_not_start_recentering(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                controller = self.controller()
                current_lane = straight_lane(offset=sign * 0.25, steer_deg=sign * 10.0)
                for _ in range(20):
                    result = self.step(controller, current_lane)
                    self.assertTrue(result.holding_straight)
                    self.assertTrue(result.enabled)
                    self.assertEqual(result.status, "simulation")
                    self.assertEqual(result.desired_steer_deg, 0)
                    self.assertEqual(result.target_steer_deg, 0)
                    self.assertEqual(result.error_deg, 0)
                    self.assertEqual(result.motor_command, 0)

    def test_straight_means_calibrated_centre_including_wrap_and_direction(self):
        cases = [(359.8, 0.2, 1), (359.8, 0.2, -1), (123.0, 123.0, -1), (0, 360, 1)]
        for centre, angle, direction in cases:
            with self.subTest(centre=centre, angle=angle, direction=direction):
                self.sensor.angle = angle
                controller = self.controller(center_deg=centre, sensor_direction=direction)
                result = self.step(controller)
                self.assertTrue(result.holding_straight)
                self.assertEqual(result.desired_steer_deg, 0)
                self.assertLessEqual(abs(result.target_steer_deg), 0.4 + 1e-8)
                self.assertEqual(result.motor_command, 0)

    def test_near_edge_or_heading_change_resumes_lane_tracking(self):
        for changes in (dict(straight_clearance=0.11), dict(straight_clearance=-0.01),
                        dict(heading_deg=3.1), dict(heading_deg=-3.1),
                        dict(way="left"), dict(way="right")):
            with self.subTest(changes=changes):
                controller = self.controller()
                self.assertTrue(self.step(controller).holding_straight)
                result = self.step(controller, straight_lane(**changes))
                self.assertTrue(result.enabled)
                self.assertFalse(result.holding_straight)
                self.assertEqual(result.desired_steer_deg, 10)
                self.assertGreater(result.target_steer_deg, 0)

    def test_measured_turn_exits_straight_mode(self):
        for angle in (1.1, 358.9):
            controller = self.controller()
            self.sensor.angle = 0
            self.assertTrue(self.step(controller).holding_straight)
            self.sensor.angle = angle
            result = self.step(controller)
            self.assertTrue(result.enabled)
            self.assertFalse(result.holding_straight)
            self.assertEqual(result.desired_steer_deg, 10)

    def test_explicit_straight_junction_remains_eligible(self):
        result = self.step(self.controller(), straight_lane(way="straight"))
        self.assertTrue(result.holding_straight)
        self.assertEqual(result.motor_command, 0)

    def test_edge_hysteresis_prevents_switching_on_small_clearance_changes(self):
        controller = self.controller(straight_edge_margin=0.15, straight_hysteresis=0.03)
        for clearance, expected in [(0.14, False), (0.16, True), (0.14, True),
                                    (0.121, True), (0.119, False), (0.14, False),
                                    (0.16, True)]:
            with self.subTest(clearance=clearance, expected=expected):
                result = self.step(controller, straight_lane(straight_clearance=clearance))
                self.assertEqual(result.holding_straight, expected)
                self.assertEqual(result.desired_steer_deg, 0 if expected else 10)

    def test_feature_can_be_disabled(self):
        controller = self.controller(keep_straight=False)
        for _ in range(100):
            result = self.step(controller)
        self.assertFalse(result.holding_straight)
        self.assertAlmostEqual(result.target_steer_deg, 10, places=3)
        self.assertGreater(result.motor_command, 0)

    def test_missing_or_invalid_geometry_falls_back_to_normal_tracking(self):
        for field in ("heading_deg", "straight_clearance"):
            for value in (None, float("nan"), float("inf"), "unknown"):
                with self.subTest(field=field, value=value):
                    controller = self.controller()
                    self.assertTrue(self.step(controller).holding_straight)
                    result = self.step(controller, straight_lane(**{field: value}))
                    self.assertTrue(result.enabled)
                    self.assertFalse(result.holding_straight)
                    self.assertEqual(result.desired_steer_deg, 10)
                    # A gap also clears the lower hysteresis entry threshold.
                    self.assertFalse(self.step(controller,
                        straight_lane(straight_clearance=0.14)).holding_straight)
        current_lane = straight_lane()
        del current_lane.heading_deg
        del current_lane.straight_clearance
        result = self.step(self.controller(), current_lane)
        self.assertTrue(result.enabled)
        self.assertFalse(result.holding_straight)
        self.assertGreater(result.target_steer_deg, 0)

    def test_vision_and_frame_faults_still_disable_the_motor(self):
        cases = [(dict(found=False), {}, "lane_lost"),
                 (dict(held=True), {}, "lane_held"),
                 (dict(obstacle=100), {}, "obstacle"),
                 (dict(steer_deg=float("nan")), {}, "lane_invalid"),
                 (dict(offset=float("nan")), {}, "lane_invalid"),
                 ({}, dict(frame_age=0.31), "stale_frame")]
        for changes, kwargs, expected in cases:
            with self.subTest(expected=expected):
                controller = self.controller()
                self.assertTrue(self.step(controller).holding_straight)
                result = self.step(controller, straight_lane(**changes), **kwargs)
                self.assertEqual(result.status, expected)
                self.assertFalse(result.enabled)
                self.assertFalse(result.holding_straight)
                self.assertEqual(result.motor_command, 0)
                self.assertEqual(self.sensor.written[-1].motor_command, 0)
                recovered = self.step(controller, straight_lane(straight_clearance=0.14))
                self.assertTrue(recovered.enabled)
                self.assertFalse(recovered.holding_straight)

    def test_sensor_faults_still_disable_the_motor(self):
        for angle, age, connected, expected in [
            (None, 0, True, "sensor_invalid"), (0, 0.2, True, "stale_sensor"),
            (0, 0, False, "sensor_disconnected"), (40, 0, True, "sensor_out_of_range")]:
            with self.subTest(expected=expected):
                self.sensor.angle, self.sensor.age, self.sensor.connected = 0, 0, True
                controller = self.controller()
                self.assertTrue(self.step(controller).holding_straight)
                self.sensor.angle, self.sensor.age, self.sensor.connected = angle, age, connected
                result = self.step(controller)
                self.assertEqual(result.status, expected)
                self.assertFalse(result.enabled)
                self.assertFalse(result.holding_straight)
                self.assertEqual(result.motor_command, 0)

    def test_disabled_mode_and_motor_failure_clear_holding_flag(self):
        result = self.step(self.controller(mode="disabled"))
        self.assertEqual(result.status, "disabled")
        self.assertFalse(result.holding_straight)

        def disconnected(*_args):
            raise OSError("Disconnected motor")

        controller = self.controller()
        self.assertTrue(self.step(controller).holding_straight)
        self.sensor.write = disconnected
        result = self.step(controller)
        self.assertEqual(result.status, "motor_disconnected")
        self.assertFalse(result.enabled)
        self.assertFalse(result.holding_straight)
        self.assertEqual(result.motor_command, 0)

    def test_hardware_tracking_status_is_preserved(self):
        result = self.step(self.controller(mode="serial", enabled=True, calibrated=True))
        self.assertEqual(result.status, "tracking")
        self.assertTrue(result.enabled)
        self.assertTrue(result.holding_straight)
        self.assertEqual(result.motor_command, 0)

    def test_entering_straight_mode_preserves_rate_and_acceleration_limits(self):
        controller = self.controller(max_rate_deg_s=12, max_accel_deg_s2=30)
        dt, previous, speed = 0.05, 0.0, 0.0
        for index in range(200):
            entering = index >= 8
            result = self.step(controller, straight_lane(heading_deg=0 if entering else 10), dt)
            new_speed = (result.target_steer_deg - previous) / dt
            self.assertLessEqual(abs(new_speed), 12 + 1e-8)
            self.assertLessEqual(abs(new_speed - speed), 30 * dt + 1e-8)
            if entering:
                self.assertTrue(result.holding_straight)
                self.assertEqual(result.desired_steer_deg, 0)
            if index == 8:
                self.assertGreater(result.target_steer_deg, 0)  # Pending motion must brake smoothly.
            previous, speed = result.target_steer_deg, new_speed
        self.assertAlmostEqual(result.target_steer_deg, 0, places=4)
        self.assertEqual(result.motor_command, 0)

    def test_invalid_straight_settings_are_rejected(self):
        invalid = [("keep_straight", "true"), ("keep_straight", 1),
                   ("straight_angle_tolerance_deg", 0), ("straight_angle_tolerance_deg", -1),
                   ("straight_angle_tolerance_deg", 30),
                   ("straight_heading_tolerance_deg", 0), ("straight_heading_tolerance_deg", 90),
                   ("straight_heading_tolerance_deg", float("nan")),
                   ("straight_edge_margin", 0), ("straight_edge_margin", -0.1),
                   ("straight_edge_margin", 0.5), ("straight_edge_margin", float("inf")),
                   ("straight_hysteresis", -0.01), ("straight_hysteresis", 0.15)]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    validate_control_config({key: value})


def road_mask(shift=0.3, heading=0.0):
    """Parallel ground edges; shift changes position, heading changes direction."""
    h, w, horizon = 180, 320, 0.33 * 180
    mask = np.zeros((h, w), dtype=np.uint8)
    for row in range(70, h):
        depth = row - horizon
        centre = (w - 1) / 2 + shift * depth + heading * h
        left = max(0, int(centre - 0.6 * depth))
        right = min(w, int(centre + 0.6 * depth) + 1)
        mask[row, left:right] = 1
    return mask


class StraightGeometryTests(unittest.TestCase):
    def test_offset_road_has_straight_heading_despite_nonzero_target_bearing(self):
        for shift in (-0.3, 0.3):
            with self.subTest(shift=shift):
                result = LaneEstimator(horizon=0.33, stabilize=False)(road_mask(shift))
                self.assertTrue(result.found)
                self.assertFalse(result.held)
                self.assertGreater(abs(result.offset), 0.2)
                self.assertGreater(abs(result.steer_deg), 3)
                self.assertLess(abs(result.heading_deg), 0.5)
                self.assertGreater(result.straight_clearance, 0.15)
                self.assertLess(result.straight_clearance, 0.35)
                clock = Clock()
                controller = SteeringController(adapter=Sensor(clock), clock=clock)
                command = controller.update(result, 1 / 30)
                self.assertTrue(command.enabled)
                self.assertTrue(command.holding_straight)
                self.assertEqual(command.motor_command, 0)

    def test_clearance_checks_far_road_even_when_vehicle_is_inside_near_edges(self):
        result = LaneEstimator(horizon=0.33, stabilize=False)(road_mask(-0.1, 0.14))
        self.assertTrue(result.found)
        cx = (320 - 1) / 2
        clearances = np.minimum(cx - result.left, result.right - cx) / (result.right - result.left)
        self.assertGreater(clearances[0], 0.15)
        self.assertLess(clearances[-1], 0)
        self.assertAlmostEqual(result.straight_clearance, float(clearances.min()), places=8)
        self.assertGreater(abs(result.heading_deg), 3)

    def test_junction_trimming_cannot_hide_the_straight_ray_leaving_the_road(self):
        estimator = LaneEstimator(horizon=0.33, stabilize=False)
        estimator.widths.extend([1.2] * WARMUP)

        def trim_to_shared_road(result, road, width, focal):
            # A junction normally leaves only the shared nearby road in these arrays.
            # That crop must not erase the risk of continuing straight beyond it.
            keep = result.rows > 150
            result.rows, result.left, result.right, result.path = (
                field[keep] for field in (result.rows, result.left, result.right, result.path))

        with patch.object(estimator, "_junction", side_effect=trim_to_shared_road) as junction:
            result = estimator(road_mask(-0.1, 0.14))
        junction.assert_called_once()
        self.assertTrue(result.found)
        cx = (320 - 1) / 2
        retained_clearances = np.minimum(cx - result.left, result.right - cx) / (result.right - result.left)
        self.assertGreater(float(retained_clearances.min()), 0.15)
        self.assertLess(result.straight_clearance, 0)


if __name__ == "__main__":
    unittest.main()
