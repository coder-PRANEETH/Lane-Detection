"""HailoRT semantic segmentation for a separately compiled, compatible HEF.

The HEF must accept RGB uint8 pixels with its normalization embedded in the
compiled graph. TorchScript weights are not executable by a Hailo device.
"""
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from .segmenter import validate_frame


class HailoSegmenter:
    def __init__(self, hef=None, input_name=None, output_name=None,
                 output_layout="HWC", output_kind="logits", road_class=1,
                 smoothing=0.0, hysteresis=0.0):
        if not hef or Path(hef).suffix.lower() != ".hef" or not Path(hef).is_file():
            raise ValueError("model.hailo.hef must point to a compiled segmentation .hef file; "
                             "the existing yolopv2.pt cannot run on Hailo")
        if output_layout not in ("HWC", "CHW"):
            raise ValueError("model.hailo.output_layout must be HWC or CHW")
        if output_kind not in ("logits", "probabilities", "class_ids"):
            raise ValueError("model.hailo.output_kind must be logits, probabilities or class_ids")
        if isinstance(road_class, bool) or not isinstance(road_class, int) or road_class < 0:
            raise ValueError("model.hailo.road_class must be a nonnegative integer")
        if not 0 <= smoothing < 1 or not 0 <= hysteresis < 0.5:
            raise ValueError("model smoothing must be in [0, 1), hysteresis in [0, 0.5)")
        try:
            import hailo_platform as hp
        except ImportError as exc:
            raise RuntimeError("The hailo backend requires HailoRT Python bindings and a matching "
                               "Pi Hailo driver/runtime; install them on Raspberry Pi OS") from exc
        self.output_layout, self.output_kind, self.road_class = output_layout, output_kind, road_class
        self.smoothing, self.hysteresis = smoothing, hysteresis
        self.device = "hailo"
        self._stack = ExitStack()
        try:
            model = hp.HEF(str(hef))
            inputs, outputs = model.get_input_vstream_infos(), model.get_output_vstream_infos()
            if len(inputs) != 1:
                raise ValueError("Hailo segmentation requires a HEF with exactly one RGB input")
            self.input_info = self._select(inputs, input_name, "input")
            self.output_info = self._select(outputs, output_name, "output")
            self.input_name, self.output_name = self.input_info.name, self.output_info.name
            self.input_shape = self._shape(self.input_info.shape, "input")
            if len(self.input_shape) != 3 or self.input_shape[2] != 3:
                raise ValueError(f"Hailo input must have H×W×3 metadata, got {self.input_shape}")
            # HEF vstream metadata is HWC; host order is set explicitly below.
            self.output_shape = self._shape(self.output_info.shape, "output")
            if len(self.output_shape) != 3:
                raise ValueError("Hailo output must be a dense H×W×C semantic segmentation tensor")
            channels = self.output_shape[2]
            if output_kind == "class_ids" and channels != 1:
                raise ValueError("class_ids output requires exactly one channel")
            if output_kind != "class_ids" and road_class >= (2 if channels == 1 else channels):
                raise ValueError(f"road_class {road_class} is outside the output's class range")
            params = hp.VDevice.create_params()
            params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.NONE
            target = self._stack.enter_context(hp.VDevice(params=params))
            groups = target.configure(model, hp.ConfigureParams.create_from_hef(
                hef=model, interface=hp.HailoStreamInterface.PCIe))
            if len(groups) != 1:
                raise ValueError("Hailo segmentation requires a HEF with one network group")
            group = groups[0]
            input_params = hp.InputVStreamParams.make_from_network_group(
                group, quantized=True, format_type=hp.FormatType.UINT8)
            output_params = hp.OutputVStreamParams.make_from_network_group(
                group, quantized=False, format_type=hp.FormatType.FLOAT32)
            input_params[self.input_name].user_buffer_format.order = hp.FormatOrder.NHWC
            output_params[self.output_name].user_buffer_format.order = (
                hp.FormatOrder.NHWC if output_layout == "HWC" else hp.FormatOrder.NCHW)
            self._pipeline = self._stack.enter_context(hp.InferVStreams(group, input_params, output_params))
            self._stack.enter_context(group.activate(group.create_params()))
        except Exception:
            self._stack.close()
            raise
        self._closed = False
        self.reset()

    @staticmethod
    def _select(infos, name, kind):
        names = [info.name for info in infos]
        if name is None and len(infos) == 1:
            return infos[0]
        if name not in names:
            raise ValueError(f"Set model.hailo.{kind}_name to one of: {names}; got {name!r}")
        return next(info for info in infos if info.name == name)

    @staticmethod
    def _shape(shape, kind):
        result = tuple(int(size) for size in shape)
        if not result or any(size <= 0 for size in result):
            raise ValueError(f"Invalid Hailo {kind} shape: {shape}")
        return result

    def reset(self):
        self.prob = self.mask = None

    def __call__(self, frame):
        return self.collect(self.submit(frame))

    def submit(self, frame):
        """HailoRT VStreams runs synchronously; retain the runner's common interface."""
        if self._closed:
            raise RuntimeError("Hailo segmenter is closed")
        validate_frame(frame)
        h, w = frame.shape[:2]
        ih, iw, _ = self.input_shape
        scale = min(iw / w, ih / h)
        rw, rh = max(1, min(iw, round(w * scale))), max(1, min(ih, round(h * scale)))
        left, top = (iw - rw) // 2, (ih - rh) // 2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        boxed = np.full((ih, iw, 3), 114, dtype=np.uint8)
        boxed[top:top + rh, left:left + rw] = rgb
        outputs = self._pipeline.infer({self.input_name: boxed[None]})
        if not isinstance(outputs, dict) or self.output_name not in outputs:
            raise ValueError(f"Hailo inference did not return output {self.output_name!r}")
        prob = self._probability(outputs[self.output_name])
        interpolation = cv2.INTER_NEAREST if self.output_kind == "class_ids" else cv2.INTER_LINEAR
        prob = cv2.resize(prob, (iw, ih), interpolation=interpolation)
        prob = prob[top:top + rh, left:left + rw]
        prob = cv2.resize(prob, (w, h), interpolation=interpolation)
        if self.prob is not None and self.prob.shape == prob.shape and self.smoothing:
            prob = self.smoothing * self.prob + (1 - self.smoothing) * prob
        self.prob = prob
        if self.mask is not None and self.mask.shape == prob.shape and self.hysteresis:
            threshold = 0.5 + self.hysteresis * (1 - 2 * self.mask.astype(np.float32))
        else:
            threshold = 0.5
        self.mask = (prob > threshold).astype(np.uint8)
        return self.mask

    def _probability(self, output):
        if not isinstance(output, np.ndarray) or not np.issubdtype(output.dtype, np.number):
            raise ValueError("Hailo segmentation output must be a numeric NumPy tensor")
        expected = self.output_shape
        if self.output_layout == "CHW":
            expected = (expected[2], expected[0], expected[1])
        if output.shape != (1, *expected):
            raise ValueError(f"Hailo output shape {output.shape} does not match expected {(1, *expected)}")
        if not np.isfinite(output).all():
            raise ValueError("Hailo segmentation output contains NaN or infinity")
        values = output[0].astype(np.float32)
        if self.output_layout == "CHW":
            values = values.transpose(1, 2, 0)
        channels = values.shape[2]
        if self.output_kind == "class_ids":
            values = values[:, :, 0]
            if np.any(values < 0) or not np.equal(values, np.rint(values)).all():
                raise ValueError("class_ids output contains noninteger or negative class labels")
            return (values == self.road_class).astype(np.float32)
        if self.output_kind == "probabilities":
            if values.min() < 0 or values.max() > 1:
                raise ValueError("probabilities output must contain values in [0, 1]")
        elif channels == 1:
            values = 1 / (1 + np.exp(-np.clip(values, -80, 80)))
        else:
            values = np.exp(values - values.max(axis=2, keepdims=True))
            values /= values.sum(axis=2, keepdims=True)
        if channels == 1:
            positive = values[:, :, 0]
            return positive if self.road_class == 1 else 1 - positive
        return values[:, :, self.road_class]

    def collect(self, pending):
        return pending

    def close(self):
        if not self._closed:
            self._closed = True
            self._stack.close()
            self.reset()
