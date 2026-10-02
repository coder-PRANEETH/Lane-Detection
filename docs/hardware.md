# Steering sensor and motor connection

`road/control.py` provides a feedback controller and a serial JSON adapter. The actual angle sensor/module and motor driver have not been specified, so this repository defines the host interface without assuming wiring, sensor registers, PWM frequency or motor-driver commands.

## Calibration

1. Start in `steering.mode="simulation"` and confirm the road overlay and right/left sign.
2. Measure the sensor reading with the steering straight; put it in `center_deg`.
3. Set `sensor_direction=1` when a right turn increases the reading, otherwise `-1`.
4. Measure allowable travel from center in encoder degrees; configure `max_steer_deg` inside those limits. A 0–360° sensor must uniquely identify every allowed steering position. A multi-turn wheel/shaft requires more information than a single wrapped reading.
5. Configure your bridge so **positive effort increases the encoder reading**, regardless of the vehicle's right/left sign. Set hardware travel limits there too.
6. Calibrate `steering_gain`, then tune `kp`, `max_command`, `max_rate_deg_s`, `max_accel_deg_s2`, and `deadband_deg` against the actual linkage. Camera bearing is only a visual target direction, not a wheel angle.
7. Check timeout, disconnect and stop behavior before setting `calibrated=true`, `enabled=true`, `mode="serial"` with a live camera source. The motor controller needs independent stop/travel protection. Steering-off is not a vehicle braking command.

`max_steer_deg` is symmetric about center and must be less than 180°. The controller rejects measured positions outside that sector and circular corrections whose shortest path would leave it.

## Serial protocol

Default connection: `/dev/ttyACM0`, 115200 baud, newline-delimited UTF-8 JSON. Install `pyserial` only for serial mode (`pip install -r requirements-hardware.txt`). A microcontroller or your existing motor module must implement these messages. A device that only transmits plain angle numbers needs an adapter for its actual protocol.

The Pi asks for a new reading:

```json
{"type":"read_angle","seq":17}
```

The bridge samples the encoder after receiving that request and replies promptly:

```json
{"type":"angle","seq":17,"angle_deg":350.0}
```

`angle_deg` must be finite and in [0, 360]. The echoed sequence must match exactly. An invalid reading can be reported as:

```json
{"type":"angle","seq":17,"valid":false}
```

The Pi discards stale/mismatched replies, rejects malformed readings and times out requests (default 30 ms). It then sends an expiring effort command, for example:

```json
{"type":"motor","enabled":true,"target_angle_deg":10.0,"delta_deg":20.0,"effort":0.35,"valid_for_ms":200}
```

`effort` is normalized [-1, 1], further bounded by `max_command`. Translate its sign and magnitude into your motor driver's control input. The bridge should apply **effort**; target and delta are telemetry. Do not independently add the delta on every message or combine it with another position loop. If the motor module accepts only position commands, implement a module-specific adapter instead.

A stopped command has `enabled=false`, `effort=0`, `valid_for_ms=0`. Startup/shutdown may send only those fields plus `type`; target and delta are optional for stop packets.

The firmware must:

- Start with the motor disabled and enforce mechanical limits.
- Accept only finite, well-formed, bounded commands.
- Disable effort when `enabled` is false, serial disconnects, or the last enabled command expires.
- Sample on each request; do not label a cached sensor value as fresh.
- Stop on sensor failures independently of the Pi.

There is no background heartbeat extending old commands while inference stalls. A Python `finally` block sends stop during ordinary exceptions/shutdown; the firmware watchdog handles host power loss, SIGKILL or an unresponsive process. The serial protocol itself does not supply board firmware.

## Feedback and diagnostics

The CSV's `measured_angle_deg` and `target_angle_deg` are absolute wrapped encoder readings. `target_steer_deg` is signed actuator displacement from calibrated center. `error_deg` is the circular target-minus-current correction. `motor_command` is normalized effort. `steer_deg` remains the smoothed camera target bearing; `offset` remains lane widths.

By default, `keep_straight` requests zero relative actuator angle when the encoder is near `center_deg`, road heading is aligned, and the forward path has sufficient edge clearance, even if the vehicle is off center. Here zero means the calibrated `center_deg` reading, which may be any absolute sensor angle. `holding_straight=true` in the CSV identifies this choice; it does not bypass motor limits or fault checks. See the [configuration details](../README.md#continuing-straight-while-off-center).

`status` explains zero effort: `disabled`, `lane_held`, `lane_lost`, `obstacle`, `stale_frame`, `stale_sensor`, `sensor_invalid`, `sensor_disconnected`, `sensor_out_of_range`, `stale_update`, `motor_disconnected`, or other explicitly rejected states. Enabled virtual feedback is labelled `simulation`; enabled physical feedback is `tracking`.

Simulated steering is a simple virtual actuator, useful for sign, wrap and continuity checks. It does not reproduce real inertia, backlash, torque or tire forces.
