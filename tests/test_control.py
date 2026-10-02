import json
import math
from types import SimpleNamespace
import unittest

from road.control import (ControlResult, SensorSample, SerialSteering, SimulatedSteering,
                          SteeringController, signed_angle_difference, validate_control_config)


def lane(angle=10.0, **changes):
    values = dict(found=True, held=False, steer_deg=angle, obstacle=None)
    values.update(changes)
    return SimpleNamespace(**values)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


class Sensor:
    def __init__(self, clock, angle=0.0):
        self.clock = clock
        self.angle = angle
        self.age = 0.0
        self.connected = True
        self.written = []
        self.closed = False

    def read(self, dt):
        return SensorSample(self.angle, self.clock() - self.age, self.connected)

    def write(self, result, timeout_s):
        self.written.append((result, timeout_s))

    def close(self):
        self.closed = True


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sensor = Sensor(self.clock)

    def controller(self, **config):
        return SteeringController(config, adapter=self.sensor, clock=self.clock)

    def step(self, controller, desired=10.0, dt=0.05, **kwargs):
        self.clock.advance(dt)
        return controller.update(lane(desired), dt, **kwargs)

    def test_circular_difference(self):
        self.assertEqual(signed_angle_difference(2, 358), 4)
        self.assertEqual(signed_angle_difference(358, 2), -4)
        self.assertEqual(signed_angle_difference(360, 0), 0)

    def test_wrap_and_sensor_direction(self):
        for direction, expected_sign in ((1, 1), (-1, -1)):
            self.sensor.angle = 359
            controller = self.controller(center_deg=359, sensor_direction=direction)
            result = None
            for _ in range(60):
                result = self.step(controller, 10)
            self.assertAlmostEqual(result.target_steer_deg, 10, places=3)
            self.assertAlmostEqual(result.target_angle_deg, (359 + direction * 10) % 360, places=3)
            self.assertAlmostEqual(result.error_deg, expected_sign * 10, places=3)
            self.assertEqual(math.copysign(1, result.motor_command), expected_sign)

    def test_rate_acceleration_and_mechanical_limits(self):
        controller = self.controller(max_rate_deg_s=12, max_accel_deg_s2=30, max_steer_deg=20)
        previous = speed = 0.0
        for desired in ([200.0] * 200 + [-200.0] * 300 + [8.0] * 200):
            result = self.step(controller, desired, dt=0.02)
            new_speed = (result.target_steer_deg - previous) / 0.02
            self.assertLessEqual(abs(result.target_steer_deg), 20)
            self.assertLessEqual(abs(new_speed), 12 + 1e-9)
            self.assertLessEqual(abs(new_speed - speed) / 0.02, 30 + 1e-7)
            self.assertLessEqual(abs(result.motor_command), 0.35)
            previous, speed = result.target_steer_deg, new_speed

    def test_rate_limits_use_actual_dt(self):
        controller = self.controller(max_rate_deg_s=8, max_accel_deg_s2=20)
        last_position = velocity = 0.0
        for dt in [0.03, 0.05, 0.1, 0.08, 0.03, 0.1] * 4:
            result = self.step(controller, 30, dt)
            new_velocity = (result.target_steer_deg - last_position) / dt
            self.assertLessEqual(abs(new_velocity), 8 + 1e-9)
            self.assertLessEqual(abs(new_velocity - velocity), 20 * dt + 1e-8)
            last_position, velocity = result.target_steer_deg, new_velocity

    def test_deadband_and_proportional_effort(self):
        controller = self.controller(deadband_deg=0.5, kp=0.1, max_command=0.2)
        first = self.step(controller, 0.2)
        self.assertTrue(first.enabled)
        self.assertEqual(first.motor_command, 0)
        for _ in range(100):
            result = self.step(controller, 10)
        self.assertEqual(result.motor_command, 0.2)

    def test_all_vision_faults_disable_motor(self):
        cases = [(lane(found=False), "lane_lost"), (lane(held=True), "lane_held"),
                 (lane(obstacle=120), "obstacle"), (lane(float("nan")), "lane_invalid"),
                 (lane(float("inf")), "lane_invalid"),
                 (lane(offset=float("nan")), "lane_invalid")]
        for current_lane, expected in cases:
            with self.subTest(expected=expected):
                controller = self.controller()
                self.step(controller)
                result = controller.update(current_lane, 0.05)
                self.assertFalse(result.enabled)
                self.assertEqual(result.motor_command, 0)
                self.assertEqual(result.status, expected)
                self.assertEqual(self.sensor.written[-1][0].motor_command, 0)

    def test_sensor_faults_disable_motor(self):
        for angle, age, connected, expected in [
            (None, 0, True, "sensor_invalid"), (float("nan"), 0, True, "sensor_invalid"),
            (361, 0, True, "sensor_invalid"), (-1, 0, True, "sensor_invalid"),
            (0, 0.2, True, "stale_sensor"), (0, -1, True, "stale_sensor"),
            (0, 0, False, "sensor_disconnected"), (50, 0, True, "sensor_out_of_range")]:
            with self.subTest(expected=expected, angle=angle):
                self.sensor.angle, self.sensor.age, self.sensor.connected = angle, age, connected
                result = self.step(self.controller())
                self.assertFalse(result.enabled)
                self.assertEqual(result.motor_command, 0)
                self.assertEqual(result.status, expected)

    def test_frame_and_update_staleness(self):
        for frame_age in [0.31, -1, float("nan")]:
            result = self.step(self.controller(), frame_age=frame_age)
            self.assertEqual(result.status, "stale_frame")
            self.assertEqual(result.motor_command, 0)
        for dt in [0, -0.1, 0.6, float("inf"), float("nan")]:
            result = self.controller().update(lane(), dt)
            self.assertEqual(result.status, "stale_update")
        controller = self.controller(mode="serial", enabled=True, calibrated=True)
        self.step(controller)
        self.clock.advance(1)
        result = controller.update(lane(), 0.05)
        self.assertEqual(result.status, "stale_update")

    def test_offline_simulation_uses_source_dt(self):
        controller = self.controller()
        self.step(controller)
        self.clock.advance(5)
        result = controller.update(lane(), 0.05)
        self.assertTrue(result.enabled)
        self.assertEqual(result.status, "simulation")

    def test_sensor_io_time_counts_into_frame_age(self):
        original_read = self.sensor.read

        def slow_read(dt):
            self.clock.advance(0.02)
            return original_read(dt)

        self.sensor.read = slow_read
        result = self.step(self.controller(), frame_age=0.29)
        self.assertEqual(result.status, "stale_frame")
        self.assertEqual(result.motor_command, 0)

    def test_recovery_starts_at_sensor_position(self):
        controller = self.controller()
        for _ in range(20):
            self.step(controller, 20)
        controller.update(lane(found=False), 0.05)
        self.sensor.angle = 2
        controller.update(lane(found=False), 0.05)
        recovered = self.step(controller, 20)
        self.assertLessEqual(recovered.target_steer_deg - 2, 120 * 0.05 ** 2 + 1e-9)

    def test_disabled_and_uncalibrated_hardware(self):
        for mode in ["disabled", "serial"]:
            controller = self.controller(mode=mode)
            result = self.step(controller)
            self.assertFalse(result.enabled)
            self.assertEqual(result.status, "disabled")
        with self.assertRaises(ValueError):
            self.controller(mode="serial", enabled=True)

    def test_motor_failure_disables_output(self):
        def disconnected(*_args):
            raise OSError("USB disconnected")
        self.sensor.write = disconnected
        result = self.step(self.controller())
        self.assertEqual(result.status, "motor_disconnected")
        self.assertFalse(result.enabled)
        self.assertEqual(result.motor_command, 0)

    def test_close_and_exception_cleanup(self):
        controller = self.controller()
        with self.assertRaisesRegex(RuntimeError, "vision failed"):
            with controller:
                self.step(controller)
                raise RuntimeError("vision failed")
        self.assertTrue(self.sensor.closed)
        self.assertFalse(self.sensor.written[-1][0].enabled)
        self.assertEqual(self.sensor.written[-1][0].motor_command, 0)
        self.assertEqual(controller.update(lane(), 0.05).status, "closed")

    def test_virtual_motor_follows_target(self):
        controller = SteeringController(clock=self.clock)
        for _ in range(200):
            result = self.step(controller, 10)
        self.assertIsInstance(controller.adapter, SimulatedSteering)
        self.assertAlmostEqual(result.measured_angle_deg, 10, delta=0.51)
        self.assertEqual(result.motor_command, 0)
        controller.close()
        self.assertEqual(controller.adapter.effort, 0)

    def test_virtual_motor_expires_watchdog(self):
        adapter = SimulatedSteering(speed_deg_s=100, clock=self.clock)
        adapter.write(ControlResult(0, 10, 10, 0.5, True, "simulation"), 0.2)
        self.assertEqual(adapter.read(0.5).angle_deg, 10)
        self.assertEqual(adapter.read(0.5).angle_deg, 10)
        self.assertEqual(adapter.effort, 0)

    def test_configuration_validation(self):
        for key, value in [("sensor_direction", 0), ("max_steer_deg", 180),
                           ("max_rate_deg_s", 0), ("max_accel_deg_s2", float("nan")),
                           ("max_command", 2), ("enabled", "false"),
                           ("center_deg", -1), ("steering_gain", -1),
                           ("serial_timeout_s", 1), ("baudrate", 9600.5)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_control_config({key: value})


class FakeSerial:
    def __init__(self, reply=None):
        self.is_open = True
        self.pending = bytearray()
        self.writes = []
        self.reply = reply or (lambda seq: [{"type": "angle", "seq": seq, "angle_deg": 359}])

    @property
    def in_waiting(self):
        return len(self.pending)

    def reset_input_buffer(self):
        self.pending.clear()

    def read(self, count):
        data = bytes(self.pending[:count])
        del self.pending[:count]
        return data

    def write(self, data):
        payload = json.loads(data)
        self.writes.append(payload)
        if payload["type"] == "read_angle":
            self.pending.extend(b"bad JSON\n")
            for response in self.reply(payload["seq"]):
                self.pending.extend((json.dumps(response) + "\n").encode())
        return len(data)

    def close(self):
        self.is_open = False


class SerialProtocolTests(unittest.TestCase):
    def test_requires_matching_sequence_and_fresh_reply(self):
        serial = FakeSerial(lambda seq: [
            {"type": "angle", "seq": seq - 1, "angle_deg": 123},
            {"type": "angle", "seq": seq, "angle_deg": 359}])
        bridge = SerialSteering("ignored", timeout_s=0.01, connection=serial)
        serial.pending.extend(b'{"type":"angle","seq":1,"angle_deg":99}\n')
        first, second = bridge.read(0.05), bridge.read(0.05)
        self.assertEqual(first.angle_deg, 359)
        self.assertEqual(second.angle_deg, 359)
        self.assertEqual([x["seq"] for x in serial.writes if x["type"] == "read_angle"], [1, 2])

    def test_stale_sequence_times_out(self):
        serial = FakeSerial(lambda seq: [{"type": "angle", "seq": seq - 1, "angle_deg": 12}])
        sample = SerialSteering("ignored", timeout_s=0.002, connection=serial).read(0.05)
        self.assertFalse(sample.connected)
        self.assertIsNone(sample.angle_deg)

    def test_faulty_reading(self):
        for response in [{"angle_deg": 999}, {"angle_deg": 1, "valid": False}, {"angle_deg": "20"}]:
            serial = FakeSerial(lambda seq: [{"type": "angle", "seq": seq, **response}])
            bridge = SerialSteering("ignored", timeout_s=0.01, connection=serial)
            self.assertIsNone(bridge.read(0.05).angle_deg)

    def test_watchdog_and_close_protocol(self):
        serial = FakeSerial()
        bridge = SerialSteering("ignored", timeout_s=0.01, connection=serial)
        self.assertFalse(serial.writes[0]["enabled"])
        bridge.write(ControlResult(359, 2, 3, 0.12, True, "tracking"), 0.2)
        command = serial.writes[-1]
        self.assertEqual(command["delta_deg"], 3)
        self.assertEqual(command["valid_for_ms"], 200)
        self.assertEqual(command["effort"], 0.12)
        bridge.close()
        self.assertFalse(serial.is_open)
        self.assertFalse(serial.writes[-1]["enabled"])
        self.assertEqual(serial.writes[-1]["valid_for_ms"], 0)
        self.assertEqual(serial.writes[-1]["effort"], 0)


if __name__ == "__main__":
    unittest.main()
