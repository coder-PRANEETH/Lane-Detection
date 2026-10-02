"""Run every frame of the four regression clips and verify logged control invariants."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from road.config import load_config, validate_config
from road.segmenter import create_segmenter
from run import process

VIDEOS = ("near_hostel", "nmv_road", "perfect_roada", "road_video")


def check_log(path, config, expected_frames):
    previous = None
    count = 0
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        count += 1
        dt = min(float(row["dt_s"]), config["lane"]["max_dt"])
        offset, steer = float(row["offset"]), float(row["steer_deg"])
        target, error, effort = (float(row[k]) for k in ("target_angle_deg", "error_deg", "motor_command"))
        assert all(math.isfinite(x) for x in (dt, offset, steer, target, error, effort)), row
        assert 0 <= target < 360, row
        assert -180 <= error < 180, row
        assert abs(effort) <= config["steering"]["max_command"] + 1e-8, row
        assert abs(float(row["target_steer_deg"])) <= config["steering"]["max_steer_deg"] + 1e-8, row
        if previous:
            assert abs(offset - previous[0]) <= config["lane"]["max_offset_rate"] * dt + 1e-8, row
            assert abs(steer - previous[1]) <= config["lane"]["max_steer_rate"] * dt + 1e-8, row
        if row["found"] == "0" or row["held"] == "1" or row["obstacle_row"]:
            assert row["enabled"] == "False" and effort == 0, row
        if row["enabled"] == "False":
            assert effort == 0, row
        if row["measured_angle_deg"]:
            measured = float(row["measured_angle_deg"])
            assert abs(error - ((target - measured + 180) % 360 - 180)) < 1e-8, row
        if row.get("holding_straight") == "True":
            steering = config["steering"]
            assert steering["keep_straight"] and row["enabled"] == "True", row
            assert row["measured_angle_deg"], row
            assert float(row["desired_steer_deg"]) == 0, row
            assert abs(float(row["heading_deg"])) <= steering["straight_heading_tolerance_deg"], row
            assert float(row["straight_clearance"]) >= steering["straight_edge_margin"] - steering["straight_hysteresis"], row
            assert abs((measured - steering["center_deg"] + 180) % 360 - 180) <= steering["straight_angle_tolerance_deg"], row
            assert row["following"] not in ("left", "right"), row
        previous = (offset, steer)
    assert count == expected_frames, f"expected {expected_frames} frames; decoded {count}"
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--videos", type=Path, default=ROOT / "videos_lowres")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "regression")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config["source"]["mode"] = "video"
    config["steering"].update(mode="simulation", enabled=False)
    config["output"].update(directory=str(args.out), max_frames=0, csv=True,
                             show=False, save_video=not args.no_save)
    validate_config(config)
    if not config["lane"]["stabilize"]:
        raise ValueError("Regression verifies smoothing: enable lane.stabilize")
    paths = [args.videos / f"{name}.mp4" for name in VIDEOS]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    segment = create_segmenter(config["model"])
    reports = []
    try:
        for path in paths:
            probe = cv2.VideoCapture(str(path))
            expected = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            probe.release()
            source = dict(config["source"], path=str(path))
            summary, _ = process(source, segment, config)
            check_log(args.out / f"{path.stem}_steering.csv", config, expected)
            if not args.no_save:
                probe = cv2.VideoCapture(str(args.out / f"{path.stem}_lane.mp4"))
                written_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
                probe.release()
                assert written_frames == expected, f"expected {expected} annotated frames; wrote {written_frames}"
            summary["checks_passed"] = True
            reports.append(summary)
            (args.out / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(f"PASS: {path.name}, all {expected} frames and control invariants", flush=True)
        (args.out / "config_used.json").write_text(json.dumps(config, indent=2) + "\n")
    finally:
        segment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
