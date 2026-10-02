"""Validated JSON configuration; all file paths are relative to the config file."""
import copy
import json
import math
from pathlib import Path


DEFAULTS = {
    "source": {"mode": "video", "path": "videos_lowres/road_video.mp4", "device": 0,
               "width": 640, "height": 360, "fps": 30.0, "read_timeout": 2.0},
    "model": {"backend": "torch", "weights": "models/yolopv2.pt", "device": None,
              "input_width": 640, "smoothing": 0.0, "hysteresis": 0.0,
              "hailo": {"hef": "models/road_segmentation.hef", "input_name": None,
                        "output_name": None, "output_layout": "HWC",
                        "output_kind": "logits", "road_class": 1}},
    "lane": {"lookahead": 0.4, "horizon": None, "hfov": 70.0, "average": 5,
             "hold_frames": 15, "row_step": 4, "prefer": "straight", "stabilize": True,
             "smoothing_tau": 0.25, "max_offset_rate": 0.35,
             "max_steer_rate": 20.0, "max_dt": 0.1},
    "steering": {"mode": "simulation", "enabled": False, "calibrated": False,
                 "center_deg": 0.0, "sensor_direction": 1, "steering_gain": 1.0,
                 "max_steer_deg": 30.0, "max_rate_deg_s": 45.0,
                 "max_accel_deg_s2": 120.0, "deadband_deg": 0.5,
                 "kp": 0.04, "max_command": 0.35,
                 "max_frame_age_s": 0.3, "max_sensor_age_s": 0.15,
                 "max_update_gap_s": 0.5, "command_timeout_s": 0.2,
                 "simulated_speed_deg_s": 90.0,
                 "serial_port": "/dev/ttyACM0", "baudrate": 115200,
                 "serial_timeout_s": 0.03, "keep_straight": True,
                 "straight_angle_tolerance_deg": 1.0, "straight_heading_tolerance_deg": 3.0,
                 "straight_edge_margin": 0.15, "straight_hysteresis": 0.03},
    "output": {"directory": "outputs", "save_video": True, "csv": True,
               "show": False, "max_frames": 0, "progress_every": 300},
}


def _merge(defaults, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'configuration'} must be an object")
    for key, value in supplied.items():
        if key not in defaults:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        if isinstance(defaults[key], dict):
            _merge(defaults[key], value, prefix + key + ".")
        else:
            defaults[key] = value


def validate_config(config):
    """Fail before opening a camera, serial port, or model for invalid settings."""
    from .control import validate_control_config

    validate_control_config(config["steering"])
    def choice(section, key, options):
        if config[section][key] not in options:
            raise ValueError(f"{section}.{key} must be one of {options}")

    def number(section, key, low, high=None, integer=False, nullable=False):
        value = config[section][key]
        if value is None and nullable:
            return
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < low
                or (high is not None and value > high)
                or (integer and not isinstance(value, int))):
            raise ValueError(f"invalid {section}.{key}: {value!r}")

    for section, keys in {"lane": ["stabilize"], "steering": ["enabled", "calibrated"],
                          "output": ["save_video", "csv", "show"]}.items():
        for key in keys:
            if not isinstance(config[section][key], bool):
                raise ValueError(f"{section}.{key} must be true or false")
    choice("source", "mode", ("video", "camera", "picamera2"))
    choice("model", "backend", ("torch", "hailo"))
    choice("lane", "prefer", ("straight", "left", "right"))
    choice("steering", "mode", ("simulation", "serial", "disabled"))
    for key in ("width", "height"):
        number("source", key, 32, 8192, integer=True)
    number("source", "fps", 0.1, 240)
    number("source", "read_timeout", 0.01, 60)
    if (not isinstance(config["source"]["device"], (int, str))
            or isinstance(config["source"]["device"], bool)):
        raise ValueError("source.device must be a camera index or device path")
    number("model", "input_width", 32, 4096, integer=True)
    if config["model"]["input_width"] % 32:
        raise ValueError("model.input_width must be divisible by 32")
    number("model", "smoothing", 0, 0.999)
    number("model", "hysteresis", 0, 0.49)
    number("lane", "lookahead", 0.01, 1)
    number("lane", "horizon", 0, 0.8, nullable=True)
    number("lane", "hfov", 1, 179)
    for key in ("average", "row_step"):
        number("lane", key, 1, 100, integer=True)
    number("lane", "hold_frames", 0, 300, integer=True)
    for key in ("smoothing_tau", "max_offset_rate", "max_steer_rate", "max_dt"):
        number("lane", key, 0.000001)
    number("output", "max_frames", 0, integer=True)
    number("output", "progress_every", 1, integer=True)
    for section, key in (("source", "path"), ("model", "weights"), ("output", "directory")):
        if not isinstance(config[section][key], str) or not config[section][key]:
            raise ValueError(f"{section}.{key} must be a nonempty path")
    h = config["model"]["hailo"]
    if h["output_layout"] not in ("HWC", "CHW"):
        raise ValueError("model.hailo.output_layout must be HWC or CHW")
    if h["output_kind"] not in ("logits", "probabilities", "class_ids"):
        raise ValueError("invalid model.hailo.output_kind")
    if not isinstance(h["hef"], str) or not h["hef"]:
        raise ValueError("model.hailo.hef must be a nonempty path")
    if isinstance(h["road_class"], bool) or not isinstance(h["road_class"], int) or h["road_class"] < 0:
        raise ValueError("model.hailo.road_class must be a nonnegative integer")
    s = config["steering"]
    if s["mode"] == "serial" and s["enabled"]:
        if config["source"]["mode"] == "video":
            raise ValueError("physical motor output requires a live camera, not a recorded video")
        if not s["calibrated"]:
            raise ValueError("set steering.calibrated only after measuring center, direction and limits")
        if not config["lane"]["stabilize"]:
            raise ValueError("physical motor output requires lane.stabilize=true")
    return config


def load_config(path):
    path = Path(path).resolve()
    with path.open() as stream:
        supplied = json.load(stream)
    config = copy.deepcopy(DEFAULTS)
    _merge(config, supplied)
    validate_config(config)
    for section, key in (("source", "path"), ("model", "weights"), ("output", "directory")):
        config[section][key] = str((path.parent / config[section][key]).resolve())
    config["model"]["hailo"]["hef"] = str((path.parent / config["model"]["hailo"]["hef"]).resolve())
    return config
