import unittest

import numpy as np

from road.filters import OneEuroFilter, RateLimitedLowPass


class RateLimitedLowPassTests(unittest.TestCase):
    def test_step_is_bounded_from_startup_and_settles(self):
        smoother = RateLimitedLowPass(tau=0.25, max_rate=[0.35, 20], initial=[0, 0])
        previous = np.zeros(2)
        for _ in range(300):
            output = smoother([0.8, 30], 1 / 30)
            self.assertTrue(np.all(np.abs(output - previous) <= np.array([0.35, 20]) / 30 + 1e-12))
            previous = output
        np.testing.assert_allclose(output, [0.8, 30], atol=1e-6)

    def test_variable_dt_and_long_stall(self):
        smoother = RateLimitedLowPass(tau=0, max_rate=2, max_dt=0.1)
        for dt in [0.01, 0.04, 0.002, 10, 0.05]:
            previous = float(smoother.x)
            output = float(smoother(100, dt))
            self.assertAlmostEqual(output - previous, 2 * min(dt, 0.1))

    def test_exponential_response_matches_across_frame_rates(self):
        outputs = []
        for fps in (10, 30, 60):
            smoother = RateLimitedLowPass(tau=0.25, max_rate=1000)
            for _ in range(fps):
                output = smoother(1, 1 / fps)
            outputs.append(float(output))
        np.testing.assert_allclose(outputs, 1 - np.exp(-4), atol=1e-12)

    def test_invalid_input_does_not_poison_state(self):
        smoother = RateLimitedLowPass()
        for value, dt in [(np.nan, 0.03), (np.inf, 0.03), (1, 0), (1, -1), (1, np.nan)]:
            with self.assertRaises(ValueError):
                smoother(value, dt)
        self.assertTrue(np.isfinite(smoother(1, 0.03)))

    def test_spike_cannot_exceed_rate_limit(self):
        smoother = RateLimitedLowPass(tau=0.25, max_rate=0.35)
        smoother(0, 1 / 30)
        peak = float(smoother(1000, 1 / 30))
        self.assertLessEqual(peak, 0.35 / 30)
        for _ in range(60):
            output = float(smoother(0, 1 / 30))
        self.assertLess(output, peak / 1000)


class OneEuroTests(unittest.TestCase):
    def test_rejects_nonfinite_input_and_zero_dt(self):
        smoother = OneEuroFilter()
        for value, dt in [(np.inf, 0.1), (1, 0), (1, np.nan)]:
            with self.assertRaises(ValueError):
                smoother(value, dt)

    def test_does_not_alias_input_or_output(self):
        smoother = OneEuroFilter()
        value = np.array([1.0, 2.0])
        result = smoother(value, 0.1)
        value[:] = 100
        result[:] = -100
        np.testing.assert_allclose(smoother.x, [1, 2])


if __name__ == "__main__":
    unittest.main()
