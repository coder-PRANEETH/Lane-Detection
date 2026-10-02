"""Drivable-area segmentation with pretrained YOLOPv2 (trained on BDD100K, no training needed)."""
import urllib.request
import warnings
from pathlib import Path

import cv2
import numpy as np

WEIGHTS_URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"


class RoadSegmenter:
    """Call it on a BGR frame to get the road mask. For more speed, `submit` a frame, do other
    work while the GPU runs, then `collect` it.

    The road probability is averaged over recent frames, and a pixel only changes between
    road and not-road once its probability is clearly past 0.5, so the mask doesn't flicker.
    Call `reset` between videos.
    """

    def __init__(self, weights="models/yolopv2.pt", device=None, input_width=640,
                 smoothing=0.0, hysteresis=0.0):
        if not isinstance(input_width, int) or input_width < 32 or input_width % 32:
            raise ValueError("model.input_width must be a positive multiple of 32")
        if not 0 <= smoothing < 1 or not 0 <= hysteresis < 0.5:
            raise ValueError("model smoothing must be in [0, 1), hysteresis in [0, 0.5)")
        # A Pi using Hailo does not need the large, unrelated PyTorch package.
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The torch backend requires PyTorch; install requirements.txt "
                               "or configure backend='hailo' with a compiled HEF") from exc
        self.torch = torch
        weights = Path(weights)
        if not weights.exists():
            weights.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading YOLOPv2 weights to {weights} ...")
            partial = weights.with_suffix(weights.suffix + ".part")
            try:
                urllib.request.urlretrieve(WEIGHTS_URL, partial)
                partial.replace(weights)
            finally:
                partial.unlink(missing_ok=True)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = self.device.type == "cuda"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch.jit.load deprecation notice
            self.model = torch.jit.load(str(weights), map_location=self.device).eval()
        if self.half:
            self.model.half()
        self.input_width = input_width
        self.smoothing = smoothing    # weight of past frames in the road probability (0 = off)
        self.hysteresis = hysteresis  # how far past 0.5 a pixel's probability must go to switch
        self.reset()

    def reset(self):
        """Forget past frames, e.g. before starting another video."""
        self.prob = self.mask = None

    def __call__(self, frame):
        """Return a uint8 mask (1 = drivable road) with the same size as the BGR frame."""
        return self.collect(self.submit(frame))

    def submit(self, frame):
        """Start segmenting a frame and return at once; the GPU keeps working in the background."""
        validate_frame(frame)
        with self.torch.inference_mode():
            return self._submit(frame)

    def _submit(self, frame):
        torch = self.torch
        h, w = frame.shape[:2]
        nh = max(2, round(h * self.input_width / w / 2) * 2)
        img = frame if (w, h) == (self.input_width, nh) else \
            cv2.resize(frame, (self.input_width, nh), interpolation=cv2.INTER_AREA)
        # The network needs sides divisible by 32: pad top and bottom like YOLOPv2's letterbox.
        pad = -nh % 32
        top = pad // 2
        img = cv2.copyMakeBorder(img, top, pad - top, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).to(self.device)
        x = (x.half() if self.half else x.float())[None] / 255
        _, seg, _ = self.model(x)  # (detections, drivable area, lane lines)

        prob = seg[0, :, top:top + nh].float().softmax(0)[1]
        if self.smoothing > 0 and self.prob is not None and self.prob.shape == prob.shape:
            prob = self.smoothing * self.prob + (1 - self.smoothing) * prob
        self.prob = prob
        if self.hysteresis > 0 and self.mask is not None and self.mask.shape == prob.shape:
            # Road stays road down to 0.5 - hysteresis; not-road needs 0.5 + hysteresis.
            mask = prob > 0.5 + self.hysteresis * (1 - 2 * self.mask.float())
        else:
            mask = prob > 0.5
        self.mask = mask.to(torch.uint8)
        return self.mask, (h, w)

    def collect(self, pending):
        """Wait for a submitted frame and return its mask."""
        mask, (h, w) = pending
        mask = mask.cpu().numpy()
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def close(self):
        """Release temporal state; the Torch model follows normal Python ownership."""
        self.reset()


def validate_frame(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
            frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
        raise ValueError("Expected a nonempty H×W×3 uint8 BGR frame")


def create_segmenter(config):
    """Select the requested inference backend without falling back to another device."""
    backend = config.get("backend", "torch")
    common = {key: config[key] for key in ("smoothing", "hysteresis") if key in config}
    if backend == "torch":
        options = {key: config[key] for key in ("weights", "device", "input_width") if key in config}
        return RoadSegmenter(**options, **common)
    if backend == "hailo":
        from .hailo_segmenter import HailoSegmenter
        return HailoSegmenter(**config.get("hailo", {}), **common)
    raise ValueError(f"Unknown model backend {backend!r}; choose 'torch' or 'hailo'")
