"""Feedback steering, independent of the vision backend and Raspberry Pi GPIO.

The encoder reports one absolute revolution, 0..360 degrees. ``center_deg`` is
its calibrated straight-ahead reading; positive lane steering means right.
``sensor_direction`` maps right to increasing (+1) or decreasing (-1) readings.
The bridge must make positive motor effort increase the encoder reading.

Serial bridge protocol (UTF-8 JSON, one object per newline):

* Host: {"type":"read_angle","seq":1}
* Bridge: {"type":"angle","seq":1,"angle_deg":359.2}
* Host: {"type":"motor","enabled":true,"target_angle_deg":2.0,
  "delta_deg":2.8,"effort":0.112,"valid_for_ms":200}

The bridge MUST acquire a new encoder reading for each request and echo its
sequence, reject invalid readings, apply its calibrated motor polarity, enforce
physical travel limits, and disable its motor when ``valid_for_ms`` expires or
USB disconnects. A reply may instead contain ``valid":false`` on sensor faults.
An enabled command renews the watchdog; this module sends no background heartbeat
that could keep an old command alive while vision stalls. The host's close path
sends enabled=false, effort=0, valid_for_ms=0. Firmware watchdog enforcement is
required: software on the Pi cannot stop a motor after Pi power/process failure.
"""

from dataclasses import dataclass, replace
import json
import math
import time


def signed_angle_difference(target, current):
    """Shortest target - current rotation in [-180, 180), including 360 wrap."""
    return (float(target) - float(current) + 180.0) % 360.0 - 180.0


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clip(value, low, high):
    return min(high, max(low, value))


@dataclass(frozen=True)
class ControlResult:
    measured_angle_deg: float | None
    target_angle_deg: float
    error_deg: float
    motor_command: float
    enabled: bool
    status: str
    desired_steer_deg: float = 0.0
    target_steer_deg: float = 0.0
    holding_straight: bool = False


@dataclass(frozen=True)
class SensorSample:
    angle_deg: float | None
    timestamp: float
    connected: bool = True


DEFAULTS = {
    "mode": "simulation", "enabled": False, "calibrated": False,
    "center_deg": 0.0, "sensor_direction": 1, "steering_gain": 1.0,
    "max_steer_deg": 30.0, "max_rate_deg_s": 45.0, "max_accel_deg_s2": 120.0,
    "deadband_deg": 0.5, "kp": 0.04, "max_command": 0.35,
    "max_frame_age_s": 0.3, "max_sensor_age_s": 0.15, "max_update_gap_s": 0.5,
    "command_timeout_s": 0.2, "simulated_speed_deg_s": 90.0,
    "serial_port": "/dev/ttyACM0", "baudrate": 115200, "serial_timeout_s": 0.03,
    "keep_straight": True, "straight_angle_tolerance_deg": 1.0,
    "straight_heading_tolerance_deg": 3.0, "straight_edge_margin": 0.15,
    "straight_hysteresis": 0.03,
}


def validate_control_config(config):
    """Validate again here so hardware limits cannot be bypassed by API callers."""
    cfg = dict(DEFAULTS)
    cfg.update(config)
    if cfg["mode"] not in {"simulation", "serial", "disabled"}:
        raise ValueError("steering.mode must be simulation, serial, or disabled")
    for key in ("enabled", "calibrated", "keep_straight"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"steering.{key} must be true or false")
    for key in DEFAULTS:
        if key in {"mode", "enabled", "calibrated", "serial_port", "keep_straight"}:
            continue
        if not _number(cfg[key]):
            raise ValueError(f"steering.{key} must be finite numeric data")
    if not 0 <= cfg["center_deg"] <= 360:
        raise ValueError("steering.center_deg must be in [0, 360]")
    if cfg["sensor_direction"] not in {-1, 1}:
        raise ValueError("steering.sensor_direction must be -1 or +1")
    if not 0 < cfg["max_steer_deg"] < 180:
        raise ValueError("steering.max_steer_deg must be between 0 and 180 (exclusive)")
    if not 0 <= cfg["deadband_deg"] < cfg["max_steer_deg"]:
        raise ValueError("steering.deadband_deg must be nonnegative and below max_steer_deg")
    if not 0 < cfg["max_command"] <= 1:
        raise ValueError("steering.max_command must be in (0, 1]")
    if not 0 < cfg["straight_angle_tolerance_deg"] < cfg["max_steer_deg"]:
        raise ValueError("steering.straight_angle_tolerance_deg must be positive and below max_steer_deg")
    if not 0 < cfg["straight_heading_tolerance_deg"] < 90:
        raise ValueError("steering.straight_heading_tolerance_deg must be between 0 and 90 degrees")
    if not 0 < cfg["straight_edge_margin"] < 0.5:
        raise ValueError("steering.straight_edge_margin must be between 0 and 0.5 lane widths")
    if not 0 <= cfg["straight_hysteresis"] < cfg["straight_edge_margin"]:
        raise ValueError("steering.straight_hysteresis must be nonnegative and below straight_edge_margin")
    for key in ("steering_gain", "max_rate_deg_s", "max_accel_deg_s2", "kp",
                "max_frame_age_s", "max_sensor_age_s", "max_update_gap_s",
                "command_timeout_s", "simulated_speed_deg_s", "serial_timeout_s"):
        if cfg[key] <= 0:
            raise ValueError(f"steering.{key} must be positive")
    if cfg["serial_timeout_s"] > min(cfg["max_sensor_age_s"], cfg["command_timeout_s"]):
        raise ValueError("steering.serial_timeout_s must not exceed sensor age or watchdog timeout")
    if cfg["command_timeout_s"] > cfg["max_update_gap_s"]:
        raise ValueError("steering.command_timeout_s must not exceed max_update_gap_s")
    if not isinstance(cfg["baudrate"], int) or cfg["baudrate"] <= 0:
        raise ValueError("steering.baudrate must be a positive integer")
    if not isinstance(cfg["serial_port"], str) or not cfg["serial_port"]:
        raise ValueError("steering.serial_port must be a nonempty string")
    if cfg["mode"] == "serial" and cfg["enabled"] and not cfg["calibrated"]:
        raise ValueError("Physical steering requires steering.calibrated=true after calibration")
    return cfg


class SimulatedSteering:
    """A virtual encoder and motor; does not import or open any hardware driver."""

    def __init__(self, center_deg=0.0, speed_deg_s=90.0, clock=time.monotonic):
        self.angle_deg = center_deg % 360.0
        self.speed_deg_s = speed_deg_s
        self.effort = 0.0
        self.valid_remaining_s = 0.0
        self.clock = clock
        self.closed = False

    def read(self, dt):
        if self.closed:
            return SensorSample(None, self.clock(), connected=False)
        active_dt = min(dt, self.valid_remaining_s)
        self.angle_deg = (self.angle_deg + self.effort * self.speed_deg_s * active_dt) % 360.0
        self.valid_remaining_s = max(0.0, self.valid_remaining_s - dt)
        if self.valid_remaining_s == 0:
            self.effort = 0.0
        return SensorSample(self.angle_deg, self.clock())

    def write(self, result, timeout_s):
        self.effort = result.motor_command if result.enabled and not self.closed else 0.0
        self.valid_remaining_s = timeout_s if result.enabled and not self.closed else 0.0

    def close(self):
        self.effort = 0.0
        self.valid_remaining_s = 0.0
        self.closed = True


class SerialSteering:
    """Bounded, sequence-matched sensor requests and expiring motor commands."""

    def __init__(self, port, baudrate=115200, timeout_s=0.03, *, clock=time.monotonic,
                 connection=None):
        self.clock = clock
        self.timeout_s = timeout_s
        self.sequence = 0
        self.closed = False
        self.buffer = bytearray()
        if connection is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("Serial steering needs pyserial: pip install pyserial") from exc
            connection = serial.Serial(port, baudrate, timeout=0, write_timeout=timeout_s)
        self.connection = connection
        # On opening/restarting the host, revoke any command left by a previous host.
        try:
            self._send({"type": "motor", "enabled": False, "effort": 0.0, "valid_for_ms": 0})
        except Exception:
            self.connection.close()
            self.closed = True
            raise

    def _send(self, payload):
        encoded = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if self.connection.write(encoded) != len(encoded):
            raise OSError("Incomplete serial steering command")

    def read(self, dt):
        if self.closed or not self.connection.is_open:
            return SensorSample(None, self.clock(), connected=False)
        self.sequence += 1
        # Purge partial and complete replies from older requests before requesting data.
        self.buffer.clear()
        self.connection.reset_input_buffer()
        started = self.clock()
        self._send({"type": "read_angle", "seq": self.sequence})
        deadline = started + self.timeout_s
        while self.clock() < deadline:
            available = self.connection.in_waiting
            if available:
                self.buffer.extend(self.connection.read(min(available, 4096)))
                if len(self.buffer) > 8192:
                    raise OSError("Oversized serial steering response")
                while b"\n" in self.buffer:
                    line, _, rest = self.buffer.partition(b"\n")
                    self.buffer = bytearray(rest)
                    try:
                        response = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(response, dict):
                        continue
                    if (response.get("type") != "angle" or
                            type(response.get("seq")) is not int or
                            response["seq"] != self.sequence):
                        continue
                    angle = response.get("angle_deg")
                    if response.get("valid", True) is not True or not _number(angle) or not 0 <= angle <= 360:
                        return SensorSample(None, started)
                    return SensorSample(float(angle) % 360.0, started)
            else:
                time.sleep(min(0.001, max(0.0, deadline - self.clock())))
        return SensorSample(None, started, connected=False)

    def write(self, result, timeout_s):
        if self.closed:
            raise OSError("Serial steering connection is closed")
        self._send({"type": "motor", "enabled": result.enabled,
                    "target_angle_deg": result.target_angle_deg,
                    "delta_deg": result.error_deg,
                    "effort": result.motor_command if result.enabled else 0.0,
                    "valid_for_ms": max(1, int(timeout_s * 1000)) if result.enabled else 0})

    def close(self):
        if self.closed:
            return
        try:
            self._send({"type": "motor", "enabled": False, "effort": 0.0, "valid_for_ms": 0})
        finally:
            self.closed = True
            self.connection.close()


class SteeringController:
    def __init__(self, config=None, *, adapter=None, clock=time.monotonic):
        self.config = validate_control_config(config or {})
        self.clock = clock
        cfg = self.config
        self.permitted = cfg["mode"] == "simulation" or (cfg["mode"] == "serial" and cfg["enabled"])
        if adapter is None:
            if cfg["mode"] == "serial" and self.permitted:
                adapter = SerialSteering(cfg["serial_port"], cfg["baudrate"], cfg["serial_timeout_s"], clock=clock)
            else:
                adapter = SimulatedSteering(cfg["center_deg"], cfg["simulated_speed_deg_s"], clock)
        self.adapter = adapter
        self.target_steer = None
        self.velocity = 0.0
        self.closed = False
        self.last_update = None
        self.last_result = None
        self.holding_straight = False

    def _can_continue_straight(self, lane, measured_steer):
        """Avoid needless centering when the vehicle is aligned inside the road.

        The centre-target bearing includes lateral displacement; use the road's
        independent heading and clearance of the forward ray through the entire
        visible fit instead. Missing geometry preserves ordinary lane tracking.
        """
        cfg = self.config
        heading = getattr(lane, "heading_deg", None)
        clearance = getattr(lane, "straight_clearance", None)
        margin = cfg["straight_edge_margin"]
        if self.holding_straight:
            margin -= cfg["straight_hysteresis"]
        return (cfg["keep_straight"] and _number(heading) and _number(clearance)
                and _number(measured_steer)
                and abs(measured_steer) <= cfg["straight_angle_tolerance_deg"]
                and abs(heading) <= cfg["straight_heading_tolerance_deg"]
                and margin <= clearance <= 0.5
                and getattr(lane, "way", "") not in {"left", "right"})

    def _move_target(self, desired, dt):
        """Bound velocity/acceleration, braking before the requested position."""
        cfg = self.config
        error = desired - self.target_steer
        acceleration = cfg["max_accel_deg_s2"]
        # Leave room for this discrete step as well as the remaining braking distance.
        braking_speed = math.sqrt((acceleration * dt) ** 2 + 2 * acceleration * abs(error)) - acceleration * dt
        requested = math.copysign(min(cfg["max_rate_deg_s"], braking_speed), error) if error else 0.0
        self.velocity += _clip(requested - self.velocity, -acceleration * dt, acceleration * dt)
        next_target = self.target_steer + self.velocity * dt
        bounded = _clip(next_target, -cfg["max_steer_deg"], cfg["max_steer_deg"])
        if bounded != next_target:
            self.velocity = (bounded - self.target_steer) / dt
        self.target_steer = bounded

    def update(self, lane, dt, frame_age=0.0):
        cfg = self.config
        now = self.clock()
        valid_dt = _number(dt) and 0 < dt <= cfg["max_update_gap_s"]
        # Recorded-video simulation follows source timestamps even when inference
        # is slower than playback. Physical steering must also observe wall time.
        gap = (cfg["mode"] == "serial" and self.last_update is not None and
               now - self.last_update > cfg["max_update_gap_s"])
        self.last_update = now
        # A bad dt must never integrate a simulation for an unbounded interval.
        read_dt = float(dt) if valid_dt else 0.0
        try:
            sample = self.adapter.read(read_dt) if not self.closed else SensorSample(None, now, False)
        except (OSError, RuntimeError):
            sample = SensorSample(None, now, False)
        measured = sample.angle_deg
        valid_sensor = _number(measured) and 0 <= measured <= 360
        measured = float(measured) % 360.0 if valid_sensor else None
        measured_steer = (signed_angle_difference(measured, cfg["center_deg"]) * cfg["sensor_direction"]
                          if valid_sensor else None)
        if self.target_steer is None:
            self.target_steer = _clip(measured_steer or 0.0, -cfg["max_steer_deg"], cfg["max_steer_deg"])
        desired_raw = getattr(lane, "steer_deg", None)
        desired = (_clip(desired_raw * cfg["steering_gain"], -cfg["max_steer_deg"], cfg["max_steer_deg"])
                   if _number(desired_raw) else self.target_steer)
        after_read = self.clock()
        sensor_age = after_read - sample.timestamp if _number(sample.timestamp) else math.inf
        total_frame_age = frame_age + max(0.0, after_read - now) if _number(frame_age) else math.inf
        if self.closed:
            status = "closed"
        elif not self.permitted:
            status = "disabled"
        elif not valid_dt or gap:
            status = "stale_update"
        elif not _number(frame_age) or frame_age < 0 or total_frame_age > cfg["max_frame_age_s"]:
            status = "stale_frame"
        elif not sample.connected:
            status = "sensor_disconnected"
        elif not valid_sensor:
            status = "sensor_invalid"
        elif sensor_age < 0 or sensor_age > cfg["max_sensor_age_s"]:
            status = "stale_sensor"
        elif abs(measured_steer) > cfg["max_steer_deg"]:
            status = "sensor_out_of_range"
        elif not getattr(lane, "found", False):
            status = "lane_lost"
        elif getattr(lane, "held", False):
            status = "lane_held"
        elif getattr(lane, "obstacle", None) is not None:
            status = "obstacle"
        elif not _number(desired_raw) or not _number(getattr(lane, "offset", 0.0)):
            status = "lane_invalid"
        else:
            status = "simulation" if cfg["mode"] == "simulation" else "tracking"
        enabled = status in {"simulation", "tracking"}
        self.holding_straight = enabled and self._can_continue_straight(lane, measured_steer)
        if enabled:
            if self.holding_straight:
                desired = 0.0
            self._move_target(desired, dt)
        else:
            self.velocity = 0.0
            # Reacquire from the actual position after a fault; do not preserve an
            # unreachable target and apply a sudden large effort when vision recovers.
            if measured_steer is not None:
                self.target_steer = _clip(measured_steer, -cfg["max_steer_deg"], cfg["max_steer_deg"])
        target = (cfg["center_deg"] + cfg["sensor_direction"] * self.target_steer) % 360.0
        error = signed_angle_difference(target, measured) if measured is not None else 0.0
        # If travel spans more than half a revolution, the shortest circular path
        # can leave the calibrated mechanical sector. Never drive that shortcut.
        if enabled and abs(self.target_steer - measured_steer) >= 180:
            enabled, status = False, "unsafe_rotation"
            self.holding_straight = False
            self.velocity = 0.0
        motor = (_clip(cfg["kp"] * error, -cfg["max_command"], cfg["max_command"])
                 if enabled and abs(error) > cfg["deadband_deg"] else 0.0)
        result = ControlResult(measured, target, error, motor, enabled, status, desired,
                               self.target_steer, self.holding_straight)
        if not self.closed:
            try:
                self.adapter.write(result, cfg["command_timeout_s"])
            except (OSError, RuntimeError):
                result = replace(result, motor_command=0.0, enabled=False, status="motor_disconnected",
                                 holding_straight=False)
                self.holding_straight = False
                self.velocity = 0.0
                try:
                    self.adapter.write(result, 0.0)
                except (OSError, RuntimeError):
                    pass  # Firmware watchdog must expire the previous command.
        self.last_result = result
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.holding_straight = False
        self.velocity = 0.0
        try:
            target = (self.config["center_deg"] + self.config["sensor_direction"] * (self.target_steer or 0.0)) % 360.0
            self.adapter.write(ControlResult(None, target, 0.0, 0.0, False, "closed"), 0.0)
        except (OSError, RuntimeError):
            pass
        finally:
            try:
                self.adapter.close()
            except (OSError, RuntimeError):
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def create_control(config=None):
    return SteeringController(config)
