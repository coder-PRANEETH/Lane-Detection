"""End-to-end orchestration using deterministic images and virtual angle feedback."""
import copy
import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import run
from road.config import DEFAULTS, load_config, validate_config
from road.control import create_control


class FakeSource:
    fps = 30.0
    name = 'test'
    is_live = False
    last_frame_time = None
    error = None

    def __init__(self, count=12, live=False):
        self.count = count
        self.is_live = live
        self.closed = False

    def read(self):
        self.last_frame_time = time.monotonic()
        self.count -= 1
        return (True, np.zeros((120, 192, 3), np.uint8)) if self.count >= 0 else (False, None)

    def release(self):
        self.closed = True


class FakeSegmenter:
    def reset(self):
        pass

    def submit(self, frame):
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.array([[82, 48], [108, 48], [181, 119], [9, 119]])], 1)
        return mask

    def collect(self, job):
        return job


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = copy.deepcopy(DEFAULTS)
        self.config['output'].update(directory=self.temp.name, save_video=False, show=False)
        self.source = FakeSource()
        self.control = create_control(self.config['steering'])

    def process(self, segment=None):
        with patch('run.open_source', return_value=self.source), patch('run.create_control', return_value=self.control):
            return run.process(self.config['source'], segment or FakeSegmenter(), self.config)

    def test_all_frames_csv_feedback_and_cleanup(self):
        report, proceed = self.process()
        self.assertEqual(report['frames'], 12)
        self.assertTrue(proceed)
        self.assertTrue(self.source.closed)
        self.assertTrue(self.control.closed)
        with (Path(self.temp.name) / 'test_steering.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(float(row['dt_s']) == 1 / 30 for row in rows))
        for row in rows:
            current, target = float(row['measured_angle_deg']), float(row['target_angle_deg'])
            self.assertAlmostEqual(float(row['error_deg']), (target - current + 180) % 360 - 180)

    def test_model_failure_stops_controller_and_closes_camera(self):
        segment = FakeSegmenter()
        segment.collect = lambda job: (_ for _ in ()).throw(RuntimeError('inference failed'))
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            self.process(segment)
        self.assertTrue(self.control.closed)
        self.assertEqual(self.control.adapter.effort, 0)
        self.assertTrue(self.source.closed)

    def test_live_disconnect_stops_controller(self):
        self.source = FakeSource(count=1, live=True)
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            self.process()
        self.assertTrue(self.control.closed)
        self.assertTrue(self.source.closed)

    def test_empty_source_fails_and_closes(self):
        self.source = FakeSource(count=0)
        with self.assertRaisesRegex(RuntimeError, 'no readable frames'):
            self.process()
        self.assertTrue(self.control.closed)
        self.assertTrue(self.source.closed)

    def test_frame_limit_does_not_skip_logged_frame(self):
        self.config['output']['max_frames'] = 3
        report, _ = self.process()
        self.assertEqual(report['frames'], 3)

    def test_frame_age_includes_lane_estimation(self):
        self.source = FakeSource(count=1, live=True)
        self.config['output']['max_frames'] = 1
        from road.lane import LaneResult
        class SlowEstimator:
            def __init__(self, **kwargs):
                pass
            def __call__(self, mask, dt):
                time.sleep(0.02)
                return LaneResult(True)
        self.control.config['max_frame_age_s'] = 0.01
        with patch('run.LaneEstimator', SlowEstimator):
            report, _ = self.process()
        self.assertEqual(report['control_statuses'], {'stale_frame': 1})


class ConfigTests(unittest.TestCase):
    def test_defaults_valid(self):
        validate_config(copy.deepcopy(DEFAULTS))

    def test_unknown_keys_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{"source":{"path":"test.mp4"}}')
            config = load_config(path)
            self.assertEqual(config['source']['path'], str(Path(directory) / 'test.mp4'))
            path.write_text('{"lane":{"smoothng_tau":0.2}}')
            with self.assertRaisesRegex(ValueError, 'unknown'):
                load_config(path)

    def test_reject_unsafe_configuration(self):
        for section, key, value in [('lane', 'average', 0), ('lane', 'max_dt', float('nan')),
                                    ('source', 'fps', 0), ('output', 'csv', 'false'),
                                    ('model', 'input_width', 641), ('steering', 'max_steer_deg', 181)]:
            config = copy.deepcopy(DEFAULTS)
            config[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_config(config)
        config = copy.deepcopy(DEFAULTS)
        config['steering'].update(mode='serial', enabled=True, calibrated=True)
        with self.assertRaisesRegex(ValueError, 'live camera'):
            validate_config(config)
        config['source']['mode'] = 'camera'
        config['lane']['stabilize'] = False
        with self.assertRaisesRegex(ValueError, 'stabilize'):
            validate_config(config)

    def test_cli_source_and_config_overrides(self):
        config = run.parse_config(['0', '--max-frames', '5', '--no-save'])
        self.assertEqual(config['source']['mode'], 'camera')
        self.assertEqual(config['source']['device'], 0)
        self.assertEqual(config['output']['max_frames'], 5)
        self.assertFalse(config['output']['save_video'])

    def test_headless_main_does_not_call_gui_cleanup(self):
        with patch('run.sources', side_effect=ValueError('no video')), patch('run.cv2.destroyAllWindows') as close:
            self.assertEqual(run.main([]), 1)
            close.assert_not_called()


if __name__ == '__main__':
    unittest.main()
