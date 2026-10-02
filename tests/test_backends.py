import queue
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from road.capture import LatestFrameSource, open_source
from road.hailo_segmenter import HailoSegmenter
from road.segmenter import create_segmenter


class Context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class FakeHailo:
    """The real HailoRT API surface, with deterministic tensor data instead of a device."""
    def __init__(self):
        self.input_infos = [types.SimpleNamespace(name="rgb", shape=(4, 4, 3))]
        self.output_infos = [types.SimpleNamespace(name="road", shape=(4, 4, 2))]
        self.output = np.zeros((1, 4, 4, 2), np.float32)
        self.output[..., 1] = 10
        self.inputs = None
        self.pipeline = Context()
        self.pipeline.infer = self.infer
        self.target = Context()
        self.target.configure = lambda *args: [self.group]
        self.group = types.SimpleNamespace(create_params=lambda: object(), activate=lambda params: Context())
        self.module = types.SimpleNamespace(
            HEF=lambda path: types.SimpleNamespace(get_input_vstream_infos=lambda: self.input_infos,
                                                  get_output_vstream_infos=lambda: self.output_infos),
            VDevice=types.SimpleNamespace(),
            HailoSchedulingAlgorithm=types.SimpleNamespace(NONE="NONE"),
            HailoStreamInterface=types.SimpleNamespace(PCIe="PCIe"),
            FormatType=types.SimpleNamespace(UINT8="UINT8", FLOAT32="FLOAT32"),
            FormatOrder=types.SimpleNamespace(NHWC="NHWC", NCHW="NCHW"),
            ConfigureParams=types.SimpleNamespace(create_from_hef=lambda **kwargs: {}),
            InputVStreamParams=types.SimpleNamespace(make_from_network_group=self.make_inputs),
            OutputVStreamParams=types.SimpleNamespace(make_from_network_group=self.make_outputs),
            InferVStreams=self.make_pipeline)

        def device(**kwargs):
            return self.target
        device.create_params = lambda: types.SimpleNamespace()
        self.module.VDevice = device

    def make_inputs(self, *args, **kwargs):
        self.input_options = kwargs
        return {"rgb": types.SimpleNamespace(user_buffer_format=types.SimpleNamespace())}

    def make_outputs(self, *args, **kwargs):
        self.output_options = kwargs
        return {"road": types.SimpleNamespace(user_buffer_format=types.SimpleNamespace())}

    def make_pipeline(self, group, input_params, output_params):
        self.input_params, self.output_params = input_params, output_params
        return self.pipeline

    def infer(self, inputs):
        self.inputs = inputs
        output = self.output
        if self.output_params["road"].user_buffer_format.order == "NCHW" and output.ndim == 4:
            output = output.transpose(0, 3, 1, 2)
        return {"road": output}


class HailoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.hef = Path(self.directory.name) / "road.hef"
        self.hef.touch()
        self.fake = FakeHailo()
        patcher = patch.dict(sys.modules, {"hailo_platform": self.fake.module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_segmenter(self, **kwargs):
        segmenter = HailoSegmenter(hef=str(self.hef), **kwargs)
        self.addCleanup(segmenter.close)
        return segmenter

    def test_letterbox_bgr_to_rgb_and_remove_padding(self):
        segmenter = self.make_segmenter()
        self.fake.output[:, (0, 3), :, 1] = -10
        frame = np.zeros((2, 4, 3), np.uint8)
        frame[:] = [20, 30, 200]
        mask = segmenter(frame)
        np.testing.assert_array_equal(mask, np.ones((2, 4), np.uint8))
        sent = self.fake.inputs["rgb"]
        self.assertEqual(sent.dtype, np.uint8)
        self.assertEqual(sent.shape, (1, 4, 4, 3))
        np.testing.assert_array_equal(sent[0, 1, 0], [200, 30, 20])
        np.testing.assert_array_equal(sent[0, 0, 0], [114, 114, 114])
        self.assertEqual(self.fake.output_options["format_type"], "FLOAT32")

    def test_chw_output_is_requested_and_decoded(self):
        segmenter = self.make_segmenter(output_layout="CHW")
        mask = segmenter(np.zeros((4, 4, 3), np.uint8))
        self.assertTrue(mask.all())
        self.assertEqual(self.fake.output_params["road"].user_buffer_format.order, "NCHW")

    def test_class_ids_and_binary_probability_output(self):
        self.fake.output_infos[0].shape = (4, 4, 1)
        self.fake.output = np.ones((1, 4, 4, 1), np.float32) * 3
        segmenter = self.make_segmenter(output_kind="class_ids", road_class=3)
        self.assertTrue(segmenter(np.zeros((4, 4, 3), np.uint8)).all())
        self.fake.output[:] = 1.5
        with self.assertRaisesRegex(ValueError, "noninteger"):
            segmenter(np.zeros((4, 4, 3), np.uint8))
        binary = self.make_segmenter(output_kind="probabilities", road_class=0)
        self.fake.output[:] = 0.2
        self.assertTrue(binary(np.zeros((4, 4, 3), np.uint8)).all())

    def test_bad_shape_nonfinite_and_invalid_probability_are_rejected(self):
        segmenter = self.make_segmenter(output_kind="probabilities")
        frame = np.zeros((4, 4, 3), np.uint8)
        for output in (np.zeros((1, 3, 4, 2)), np.full((1, 4, 4, 2), np.nan),
                       np.full((1, 4, 4, 2), 1.1)):
            self.fake.output = output
            with self.assertRaises(ValueError):
                segmenter(frame)
        for bad_frame in (None, np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32)):
            with self.assertRaises(ValueError):
                segmenter(bad_frame)

    def test_smoothing_reset_and_close(self):
        segmenter = self.make_segmenter(smoothing=0.8)
        frame = np.zeros((4, 4, 3), np.uint8)
        self.assertTrue(segmenter(frame).all())
        self.fake.output[..., 1] = -10
        self.assertTrue(segmenter(frame).all())
        segmenter.reset()
        self.assertFalse(segmenter(frame).any())
        segmenter.close()
        self.assertTrue(self.fake.pipeline.closed)
        self.assertTrue(self.fake.target.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            segmenter(frame)

    def test_output_name_required_when_multiple_heads(self):
        self.fake.output_infos.append(types.SimpleNamespace(name="lanes", shape=(4, 4, 2)))
        with self.assertRaisesRegex(ValueError, "output_name"):
            self.make_segmenter()

    def test_torchscript_is_not_accepted_as_hef(self):
        with self.assertRaisesRegex(ValueError, "compiled"):
            HailoSegmenter(hef="models/yolopv2.pt")


class CaptureTests(unittest.TestCase):
    def make_live(self, timeout=0.05):
        frames = queue.Queue()
        closed = threading.Event()
        source = LatestFrameSource(frames.get, closed.set, "mock", 30, timeout)
        self.addCleanup(lambda: frames.put((False, None)))
        self.addCleanup(source.release)
        return source, frames, closed

    def test_live_drops_old_frames_and_keeps_capture_timestamp(self):
        source, frames, closed = self.make_live()
        before = time.monotonic()
        for value in range(3):
            frames.put((True, np.full((2, 2, 3), value, np.uint8)))
        with source._condition:
            self.assertTrue(source._condition.wait_for(lambda: source._sequence == 3, timeout=1))
        ok, latest = source.read()
        self.assertTrue(ok)
        self.assertTrue((latest == 2).all())
        self.assertGreaterEqual(source.last_frame_time, before)
        self.assertLessEqual(source.last_frame_time, time.monotonic())
        ok, _ = source.read()
        self.assertFalse(ok)
        self.assertIn("timed out", source.error)

    def test_disconnect_rejects_even_an_unread_last_frame(self):
        source, frames, closed = self.make_live()
        frames.put((True, np.zeros((2, 2, 3), np.uint8)))
        frames.put((False, None))
        self.assertTrue(closed.wait(1))
        self.assertEqual(source.read(), (False, None))
        self.assertIn("disconnected", source.error)

    def test_release_returns_when_a_camera_driver_is_blocked(self):
        source, frames, closed = self.make_live(timeout=0.02)
        start = time.monotonic()
        source.release()
        self.assertLess(time.monotonic() - start, 0.3)
        frames.put((False, None))
        self.assertTrue(closed.wait(1))

    def test_nan_fps_falls_back_and_failed_open_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.touch()
            from unittest.mock import Mock
            cap = Mock()
            cap.isOpened.return_value = True
            cap.get.return_value = float("nan")
            with patch("road.capture.cv2.VideoCapture", return_value=cap):
                source = open_source({"mode": "video", "path": str(path), "fps": 25})
                self.assertEqual(source.fps, 25)
                source.release()
                cap.isOpened.return_value = False
                with self.assertRaisesRegex(RuntimeError, "Cannot open"):
                    open_source({"mode": "video", "path": str(path)})
                self.assertEqual(cap.release.call_count, 2)

    def test_missing_dependency_and_invalid_config_errors(self):
        with patch.dict(sys.modules, {"picamera2": None}):
            with self.assertRaisesRegex(RuntimeError, "python3-picamera2"):
                open_source({"mode": "picamera2"})
        for config in ({"mode": "bad"}, {"fps": float("nan")},
                       {"mode": "camera", "width": 3.5}, {"mode": "video"}):
            with self.assertRaises(ValueError):
                open_source(config)

    def test_camera_device_path_has_a_safe_output_name(self):
        from unittest.mock import Mock
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30
        def latest(read, release, name, fps, timeout):
            return types.SimpleNamespace(name=name, fps=fps, release=release)
        with patch("road.capture.cv2.VideoCapture", return_value=cap), \
                patch("road.capture.LatestFrameSource", side_effect=latest):
            source = open_source({"mode": "camera", "device": "/dev/video0"})
            self.assertEqual(source.name, "cameravideo0")
            source.release()

    def test_picamera2_requests_opencv_byte_order_and_new_frames(self):
        from unittest.mock import Mock
        camera = Mock()
        frame = np.full((2, 4, 3), [20, 30, 200], np.uint8)
        camera.capture_array.return_value = frame
        constructor = Mock(return_value=camera)
        def latest(read, release, name, fps, timeout):
            return types.SimpleNamespace(read=read, release=release, name=name, fps=fps)
        with patch.dict(sys.modules, {"picamera2": types.SimpleNamespace(Picamera2=constructor)}), \
                patch("road.capture.LatestFrameSource", side_effect=latest):
            source = open_source({"mode": "picamera2", "device": 1, "width": 4, "height": 2})
            constructor.assert_called_once_with(camera_num=1)
            camera.create_video_configuration.assert_called_once_with(
                main={"size": (4, 2), "format": "RGB888"}, controls={"FrameRate": 30.0},
                buffer_count=4, queue=False)
            camera.start.assert_called_once()
            ok, captured = source.read()
            self.assertTrue(ok)
            np.testing.assert_array_equal(captured, frame)
            source.release()
            camera.close.assert_called_once()


class OptionalImportTests(unittest.TestCase):
    def test_import_road_does_not_import_torch_or_hailo(self):
        result = subprocess.run([sys.executable, "-c", "import road, sys; "
                                 "assert 'torch' not in sys.modules; "
                                 "assert 'hailo_platform' not in sys.modules"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_backend_does_not_silently_fall_back(self):
        with self.assertRaisesRegex(ValueError, "Unknown model backend"):
            create_segmenter({"backend": "automatic"})


if __name__ == "__main__":
    unittest.main()
