"""Recorded and live BGR capture, with one replaceable frame for live sources."""
import math
import threading
import time
from pathlib import Path

import cv2

from .segmenter import validate_frame


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"source.{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"source.{name} must be a finite positive number")
    return value


def _fps(value, fallback):
    return float(value) if math.isfinite(value) and value > 0 else fallback


class VideoSource:
    is_live = False

    def __init__(self, path, fallback_fps=30):
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video does not exist or is not a file: {path}")
        self.name = path.stem
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps = _fps(self._cap.get(cv2.CAP_PROP_FPS), fallback_fps)
        self.last_frame_time = None
        self.error = None

    def read(self):
        ok, frame = self._cap.read()
        if ok:
            validate_frame(frame)
            self.last_frame_time = time.monotonic()
        return ok, frame

    def release(self):
        self._cap.release()


class LatestFrameSource:
    """Read a device continuously; slow inference never builds a queue of old frames.

    Backend access and cleanup stay in the worker: calling OpenCV release() from a
    second thread while read() is active can crash some video backends. A stuck
    driver cannot block shutdown; the daemon cleans up when its read returns.
    """
    is_live = True

    def __init__(self, read_frame, release_device, name, fps, read_timeout=2.0):
        self.name, self.fps = name, fps
        self.last_frame_time = None
        self.error = None
        self._read_frame, self._release_device = read_frame, release_device
        self._timeout = _positive(read_timeout, "read_timeout")
        self._condition = threading.Condition()
        self._stopped = False
        self._finished = False
        self._frame = None
        self._timestamp = None
        self._sequence = self._consumed = 0
        self._thread = threading.Thread(target=self._capture, name=f"capture-{name}", daemon=True)
        self._thread.start()

    def _capture(self):
        try:
            while not self._stopped:
                ok, frame = self._read_frame()
                timestamp = time.monotonic()
                if not ok:
                    raise RuntimeError(f"Camera {self.name} disconnected or failed to read a frame")
                validate_frame(frame)
                with self._condition:
                    self._frame = frame.copy()
                    self._timestamp = timestamp
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception as exc:
            self.error = str(exc)
        finally:
            with self._condition:
                self._finished = True
                self._condition.notify_all()
            try:
                self._release_device()
            except Exception as exc:
                self.error = self.error or str(exc)

    def read(self):
        deadline = time.monotonic() + self._timeout
        with self._condition:
            while self._sequence == self._consumed and not self._finished and not self._stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.error = f"Camera {self.name} timed out after {self._timeout:g}s"
                    self._stopped = True
                    return False, None
                self._condition.wait(remaining)
            # A disconnected camera must stop control even if an unread frame remains.
            if self._finished or self._stopped:
                return False, None
            self._consumed = self._sequence
            self.last_frame_time = self._timestamp
            return True, self._frame

    def release(self):
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=min(self._timeout, 0.5))


def open_source(config):
    """Open one source. Folder expansion is the caller's responsibility."""
    mode = config.get("mode", "video")
    fps = _positive(config.get("fps", 30), "fps")
    if mode == "video":
        if not config.get("path"):
            raise ValueError("source.path is required for video mode")
        return VideoSource(config["path"], fps)
    if mode not in ("camera", "picamera2"):
        raise ValueError(f"Unknown source mode {mode!r}; choose video, camera or picamera2")
    width = _positive(config.get("width", 640), "width")
    height = _positive(config.get("height", 360), "height")
    if not width.is_integer() or not height.is_integer():
        raise ValueError("source.width and source.height must be whole pixels")
    width, height = int(width), int(height)
    timeout = _positive(config.get("read_timeout", 2), "read_timeout")
    device = config.get("device", 0)
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    if mode == "camera":
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera {device!r}")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            name = f"camera{Path(str(device)).name}"
            return LatestFrameSource(cap.read, cap.release, name,
                                     _fps(cap.get(cv2.CAP_PROP_FPS), fps), timeout)
        except Exception:
            cap.release()
            raise
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError("picamera2 mode requires Raspberry Pi OS python3-picamera2 "
                           "(use a virtual environment with --system-site-packages)") from exc
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise ValueError("source.device must be a nonnegative camera index for picamera2")
    camera = Picamera2(camera_num=device)
    try:
        # Picamera2's RGB888 is B,G,R byte order, matching OpenCV's BGR input.
        settings = camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps}, buffer_count=4, queue=False)
        camera.configure(settings)
        camera.start()
        return LatestFrameSource(lambda: (True, camera.capture_array("main")),
                                 camera.close, f"picamera{device}", fps, timeout)
    except Exception:
        camera.close()
        raise
