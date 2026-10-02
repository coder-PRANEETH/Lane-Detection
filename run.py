"""Camera/video -> road mask -> smoothed lane -> measured-angle steering feedback."""
import argparse
import csv
import json
import math
import signal
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2

from road import LaneEstimator, draw
from road.capture import open_source
from road.config import load_config, validate_config
from road.control import create_control
from road.segmenter import create_segmenter

ROOT = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def sources(config):
    """Expand a video directory, preserving camera device paths as camera sources."""
    source = config["source"]
    if source["mode"] != "video":
        return [source.copy()]
    path = Path(source["path"])
    if not path.exists():
        raise FileNotFoundError(f"video source does not exist: {path}")
    paths = sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"no videos found in {path}")
    return [dict(source, path=str(p)) for p in paths]


class Statistics:
    def __init__(self):
        self.frames = self.found = self.held = self.enabled = 0
        self.holding_straight = 0
        self.statuses = Counter()
        self.last = None
        self.maximum = {key: 0.0 for key in ("offset_step", "steer_step_deg", "raw_offset_step",
                                           "raw_steer_step_deg", "offset_rate", "steer_rate_deg_s")}

    def add(self, lane, control, dt):
        values = (lane.offset, lane.steer_deg, lane.raw_offset, lane.raw_steer_deg)
        if not all(math.isfinite(x) for x in values):
            raise RuntimeError("lane estimator returned a non-finite output")
        if self.last is not None:
            steps = [abs(a - b) for a, b in zip(values, self.last)]
            for key, val in zip(self.maximum, steps + [steps[0] / dt, steps[1] / dt]):
                self.maximum[key] = max(self.maximum[key], val)
        self.last = values
        self.frames += 1
        self.found += lane.found
        self.held += lane.held
        self.enabled += control.enabled
        self.holding_straight += control.holding_straight
        self.statuses[control.status] += 1


def process(source, segment, config):
    """Run a source to EOF (or configured limit); always stop and close the controller."""
    cap = control = writer = log_file = None
    stats = Statistics()
    start = time.perf_counter()
    user_quit = False
    try:
        cap = open_source(source)
        control = create_control(config["steering"])
        estimate = LaneEstimator(fps=cap.fps, **config["lane"])
        segment.reset()
        out = Path(config["output"]["directory"])
        out.mkdir(parents=True, exist_ok=True)
        name = cap.name
        output = config["output"]
        log = None
        if output["csv"]:
            log_file = (out / f"{name}_steering.csv").open("w", newline="")
            columns = ["frame", "time_s", "dt_s", "found", "held", "steer_deg", "offset",
                       "raw_steer_deg", "raw_offset", "obstacle_row", "ways", "following",
                       "frame_age_s", "measured_angle_deg", "target_angle_deg", "error_deg",
                       "motor_command", "enabled", "status", "desired_steer_deg", "target_steer_deg",
                       "heading_deg", "straight_clearance", "holding_straight"]
            log = csv.DictWriter(log_file, fieldnames=columns)
            log.writeheader()
        frame_period = 1.0 / cap.fps
        last_control_time = None
        ok, frame = cap.read()
        captured_at = first_capture = cap.last_frame_time
        if not ok:
            raise RuntimeError(f"no readable frames from {name}")
        job = segment.submit(frame)
        while job is not None:
            mask = segment.collect(job)
            # File playback has no stale-frame risk; overlap its next inference with geometry.
            next_frame = next_job = None
            limit_reached = output["max_frames"] and stats.frames + 1 >= output["max_frames"]
            if not cap.is_live and not limit_reached:
                ok, next_frame = cap.read()
                if ok:
                    next_job = segment.submit(next_frame)
            now = time.monotonic()
            dt = frame_period if not cap.is_live or last_control_time is None else now - last_control_time
            last_control_time = now
            res = estimate(mask, dt=dt)
            age = max(0.0, time.monotonic() - captured_at) if cap.is_live else 0.0
            command = control.update(res, dt, frame_age=age)
            stats.add(res, command, min(dt, config["lane"]["max_dt"]))
            if log:
                row = {"frame": stats.frames - 1,
                       "time_s": (stats.frames - 1) * frame_period if not cap.is_live else now - first_capture,
                       "dt_s": dt, "found": int(res.found), "held": int(res.held),
                       "steer_deg": res.steer_deg, "offset": res.offset,
                       "raw_steer_deg": res.raw_steer_deg, "raw_offset": res.raw_offset,
                       "obstacle_row": "" if res.obstacle is None else res.obstacle,
                       "ways": "+".join(way.name for way in res.ways), "following": res.way,
                       "frame_age_s": age, "heading_deg": res.heading_deg,
                       "straight_clearance": res.straight_clearance, **asdict(command)}
                log.writerow(row)
            elapsed = max(time.perf_counter() - start, 1e-6)
            if output["save_video"] or output["show"]:
                vis = draw(frame, res, stats.frames / elapsed, control=command)
                if output["save_video"]:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(str(out / f"{name}_lane.mp4"),
                                                cv2.VideoWriter_fourcc(*"mp4v"), cap.fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot create video writer in {out}")
                    writer.write(vis)
                if output["show"]:
                    cv2.imshow("lane", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        user_quit = True
                        break
            if stats.frames % output["progress_every"] == 0:
                print(f"{name}: {stats.frames} frames, {stats.frames / elapsed:.1f} processing fps", flush=True)
            if limit_reached:
                break
            if cap.is_live:
                ok, next_frame = cap.read()
                captured_at = cap.last_frame_time
                if not ok:
                    raise RuntimeError(f"camera disconnected or timed out: {name}: {cap.error or 'read failed'}")
                next_job = segment.submit(next_frame)
            frame, job = next_frame, next_job
        elapsed = max(time.perf_counter() - start, 1e-6)
        summary = {"source": name, "frames": stats.frames, "found_frames": stats.found,
                   "held_frames": stats.held, "fresh_lane_frames": stats.found - stats.held,
                   "control_enabled_frames": stats.enabled, "control_statuses": dict(stats.statuses),
                   "holding_straight_frames": stats.holding_straight,
                   "processing_fps": stats.frames / elapsed, "elapsed_s": elapsed,
                   "source_fps": cap.fps, "live": cap.is_live, "steering_mode": config["steering"]["mode"],
                   **stats.maximum}
        (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{name}: {stats.frames} frames complete, fresh road {(stats.found - stats.held) / stats.frames:.1%}, "
              f"{stats.frames / elapsed:.1f} fps; max offset step {stats.maximum['offset_step']:.5f}, "
              f"max steering step {stats.maximum['steer_step_deg']:.3f} deg", flush=True)
        return summary, not user_quit
    finally:
        # Stop before touching video/UI cleanup, including on exceptions and Ctrl-C.
        try:
            if control is not None:
                control.close()
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if log_file is not None:
                log_file.close()


def parse_config(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", nargs="?", help="override config: video, folder, or camera index")
    p.add_argument("--config", type=Path, default=ROOT / "config.json")
    for flag in ("lookahead", "horizon", "hfov"):
        p.add_argument("--" + flag, type=float)
    p.add_argument("--prefer", choices=["straight", "left", "right"])
    p.add_argument("--average", type=int)
    p.add_argument("--no-stabilize", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--csv", action="store_true")
    p.add_argument("--max-frames", type=int, help="0 = entire recording / continuous camera")
    p.add_argument("--weights", type=Path)
    p.add_argument("--device", help="torch device: cpu, cuda, etc.")
    args = p.parse_args(argv)
    config = load_config(args.config)
    if args.source is not None:
        if args.source.isdigit():
            config["source"].update(mode="camera", device=int(args.source))
        else:
            config["source"].update(mode="video", path=str(Path(args.source).resolve()))
    for key in ("lookahead", "horizon", "hfov", "prefer", "average"):
        if getattr(args, key) is not None:
            config["lane"][key] = getattr(args, key)
    if args.no_stabilize:
        config["lane"]["stabilize"] = False
    for key, value in (("show", args.show), ("csv", args.csv)):
        if value:
            config["output"][key] = True
    if args.no_save:
        config["output"]["save_video"] = False
    if args.out is not None:
        config["output"]["directory"] = str(args.out.resolve())
    if args.max_frames is not None:
        config["output"]["max_frames"] = args.max_frames
    if args.weights is not None:
        config["model"]["weights"] = str(args.weights.resolve())
    if args.device is not None:
        config["model"]["device"] = args.device
    return validate_config(config)


def main(argv=None):
    segment = None
    config = None
    try:
        config = parse_config(argv)
        inputs = sources(config)
        segment = create_segmenter(config["model"])
        print(f"model: {config['model']['backend']} on {segment.device}; steering: {config['steering']['mode']}", flush=True)
        for source in inputs:
            _, keep_going = process(source, segment, config)
            if not keep_going:
                break
        return 0
    except KeyboardInterrupt:
        print("stopped; motor output disabled", flush=True)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", flush=True)
        return 1
    finally:
        if segment is not None and hasattr(segment, "close"):
            segment.close()
        if config is not None and config["output"]["show"]:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
