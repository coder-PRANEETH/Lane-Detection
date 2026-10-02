"""Time-based smoothing for noisy per-frame values."""
import numpy as np


class OneEuroFilter:
    """The 1-euro filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass filter whose cutoff
    frequency rises with how fast the value is changing. A steady value gets heavy smoothing
    (little jitter); a value that really moves gets light smoothing (little lag).

    Works on a scalar or a numpy array, with a min_cutoff and beta per element if wanted.
    min_cutoff (Hz) sets the smoothing when steady; beta sets how quickly speed opens it up.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = np.asarray(min_cutoff, float)
        self.beta = np.asarray(beta, float)
        self.d_cutoff = d_cutoff
        if (not np.all(np.isfinite(self.min_cutoff)) or np.any(self.min_cutoff <= 0)
                or not np.all(np.isfinite(self.beta)) or np.any(self.beta < 0)
                or not np.isfinite(d_cutoff) or d_cutoff <= 0):
            raise ValueError("filter cutoffs must be positive and beta must be nonnegative")
        self.reset()

    def reset(self):
        self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if not np.all(np.isfinite(x)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("filter values must be finite and dt must be positive")
        if self.x is None or np.shape(self.x) != np.shape(x):
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return self.x.copy()
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


class RateLimitedLowPass:
    """Exponential low-pass with an absolute change limit per second.

    Unlike an adaptive cutoff alone, a large innovation cannot open this filter
    enough to make the output jump. Initialize at neutral for a bounded startup;
    keep this state across tracking loss. ``max_dt`` prevents a long processing
    stall from spending the entire accumulated rate budget in one command.
    """

    def __init__(self, tau=0.25, max_rate=1.0, initial=0.0, max_dt=0.1):
        self.tau = float(tau)
        self.max_rate = np.asarray(max_rate, float)
        self.max_dt = float(max_dt)
        if (not np.isfinite(self.tau) or self.tau < 0
                or not np.all(np.isfinite(self.max_rate)) or np.any(self.max_rate <= 0)
                or not np.isfinite(self.max_dt) or self.max_dt <= 0):
            raise ValueError("tau must be nonnegative; max_rate and max_dt must be positive")
        self.reset(initial)

    def reset(self, initial=0.0):
        value = np.asarray(initial, float)
        if not np.all(np.isfinite(value)):
            raise ValueError("initial filter state must be finite")
        self.x = value.copy()

    def __call__(self, value, dt):
        value = np.asarray(value, float)
        if (value.shape != self.x.shape or not np.all(np.isfinite(value))
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError("filter value must have the original shape and finite values; dt must be positive")
        dt = min(float(dt), self.max_dt)
        alpha = -np.expm1(-dt / self.tau) if self.tau else 1.0
        delta = alpha * (value - self.x)
        self.x = self.x + np.clip(delta, -self.max_rate * dt, self.max_rate * dt)
        return self.x.copy()
