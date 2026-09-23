"""Adaptive smoothing for noisy per-frame values."""
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
        self.reset()

    def reset(self):
        self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self.x is None or np.shape(self.x) != np.shape(x):
            self.x, self.dx = x, np.zeros_like(x)
            return x
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x
