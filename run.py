"""Find the drivable lane on unmarked roads, from a video, a folder of videos or a live camera.

    python run.py videos_lowres/road_video.mp4    # writes outputs/road_video_lane.mp4
    python run.py videos_lowres                   # every video in the folder
    python run.py 0 --show                        # live from camera 0, press q to quit
    python run.py clip.mp4 --csv                  # also log steering per frame

The whole road is treated as one lane: a straight strip fitted to the road the model sees,
taken as the median of the last few frames. Steering is the direction of a lookahead point on its centre.
Where the road splits (side road, T-junction, crossroad) every way to go gets its own
rectangle, and the vehicle follows the --prefer one.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2

from road import LaneEstimator, RoadSegmenter, draw

ROOT = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def sources(arg):
    if arg.isdigit():
        return [int(arg)]
    path = Path(arg)
    if path.is_dir():
        return sorted(f for f in path.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    return [path]


def process(src, segment, args):
    """Run one video or camera. Returns False if the user pressed q to stop everything."""
    cap = cv2.VideoCapture(src if isinstance(src, int) else str(src))
    if not cap.isOpened():
        print(f"can't open {src}")
        return True
    name = f"camera{src}" if isinstance(src, int) else src.stem
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    estimate = LaneEstimator(args.lookahead, args.horizon, args.hfov, args.average, prefer=args.prefer,
                             fps=fps_in, stabilize=not args.no_stabilize)
    segment.reset()
    args.out.mkdir(parents=True, exist_ok=True)

    writer = None
    log_file = open(args.out / f"{name}_steering.csv", "w", newline="") if args.csv else None
    log = csv.writer(log_file) if log_file else None
    if log:
        log.writerow(["frame", "found", "steer_deg", "offset", "obstacle_row", "ways", "following"])

    frames = found = 0
    fps = None
    keep_going = True
    ok, frame = cap.read()
    job = segment.submit(frame) if ok else None
    t = time.perf_counter()
    while job is not None:
        mask = segment.collect(job)  # waits for the GPU
        # Start the GPU on the next frame, then fit and draw this one while it runs.
        ok, upcoming = cap.read()
        job = segment.submit(upcoming) if ok else None
        res = estimate(mask)
        now = time.perf_counter()
        fps = 1 / (now - t) if fps is None else 0.9 * fps + 0.1 / (now - t)
        t = now
        vis = draw(frame, res, fps)

        if not args.no_save:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(args.out / f"{name}_lane.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
            writer.write(vis)
        if log:
            log.writerow([frames, int(res.found), f"{res.steer_deg:.2f}", f"{res.offset:.3f}",
                          "" if res.obstacle is None else int(res.obstacle),
                          "+".join(way.name for way in res.ways), res.way])
        if args.show:
            cv2.imshow("lane", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                keep_going = False
                break
        frames += 1
        found += res.found
        frame = upcoming

    cap.release()
    if writer:
        writer.release()
    if log_file:
        log_file.close()
    if frames:
        print(f"{name}: {frames} frames, road found in {found / frames:.0%}, {fps:.1f} fps")
    return keep_going


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video file, folder of videos, or camera index (0, 1, ...)")
    p.add_argument("--lookahead", type=float, default=0.4,
                   help="steering target row, 0 = horizon, 1 = bottom of the frame (default 0.4)")
    p.add_argument("--horizon", type=float,
                   help="horizon row as a fraction of image height, for a fixed camera "
                        "(default: estimated from the road edges)")
    p.add_argument("--hfov", type=float, default=70,
                   help="camera horizontal field of view in degrees, used for the steering angle (default 70)")
    p.add_argument("--prefer", choices=["straight", "left", "right"], default="straight",
                   help="way to take where the road splits, if it's there (default straight)")
    p.add_argument("--no-stabilize", action="store_true",
                   help="turn off the adaptive smoothing of the lane and steering")
    p.add_argument("--average", type=int, default=5,
                   help="take the road as the median of the last N frames, to stop it jumping (default 5)")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    p.add_argument("--show", action="store_true", help="show a live window")
    p.add_argument("--no-save", action="store_true", help="don't write the annotated video")
    p.add_argument("--csv", action="store_true", help="also write per-frame steering to a CSV")
    p.add_argument("--weights", type=Path, default=ROOT / "models" / "yolopv2.pt")
    p.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    args = p.parse_args()

    segment = RoadSegmenter(args.weights, args.device)
    print(f"model on {segment.device}")
    for src in sources(args.source):
        if not process(src, segment, args):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
