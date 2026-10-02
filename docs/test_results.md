# Test results — 2026-09-30

## Follow-up: keep a straight course while off center

The added keep-straight policy passed **81 automated tests** (including 17 new tests for off-center travel, independent heading, edge clearance, hysteresis, calibrated zero/wrap, selected turns, fault precedence and bounded transitions). A fresh **300-frame smoke run on each of the four recordings** also passed the smoothing and control checks: 1,200 frames total, with updated annotated videos and CSVs in `outputs/straight_smoke/`. The policy activated on 22 frames of `perfect_roada` and zero frames in the other three sampled prefixes. Synthetic offset-road tests verify zero motor effort on both sides of the center when the wheels and road are straight.

The full-recording results below are from the earlier smoothing/control implementation, before this follow-up policy. They are retained as that run's evidence; the follow-up did not rerun all 22,651 frames.

## Initial full-recording validation

Executed on this Linux laptop with Python 3.11.9, OpenCV 5.0.0, PyTorch 2.14.0+cu130 and an NVIDIA RTX 3050 Laptop GPU. The supplied `models/yolopv2.pt` ran at its default 640-pixel input width. Steering used simulated encoder feedback.

## Automated tests

`python -m unittest discover -s tests -v`: **64 tests passed**. Coverage includes smoothing, outlier confirmation, loss/reacquisition, timing, encoder wrap, motor limits, serial faults/watchdog, live capture buffering, mocked Picamera2/Hailo and pipeline cleanup. A separate CPU-only smoke run successfully processed one real frame (not a CPU throughput benchmark).

## Full recordings

`python scripts/test_videos.py` processed every frame of all four requested recordings. CSV row counts and saved annotated-video frame counts match the original recording metadata. All finite-value, smoothing-rate, circular-error, motor-effort, travel-limit and fault-disable assertions passed. Final controller code was also replayed against every recorded row: all eight control output fields matched exactly; enabled actuator target rate and acceleration stayed within 45 degrees/s and 120 degrees/s².

| Recording | Frames | Processing fps | Fresh lane | Virtual control enabled | Max offset step (lane widths) | Max steering step (degrees) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| near_hostel.mp4 | 7,457 | 43.8 | 53.4% | 47.7% | 0.01167 | 0.667 |
| nmv_road.mp4 | 8,278 | 40.7 | 86.9% | 57.9% | 0.01167 | 0.667 |
| perfect_roada.mp4 | 3,543 | 42.7 | 89.2% | 83.9% | 0.01167 | 0.667 |
| road_video.mp4 | 3,373 | 37.9 | 89.3% | 63.1% | 0.01167 | 0.667 |

Total: **22,651 frames**, all at 30 fps in source time. The default bounds are 0.35 lane widths/s and 20 degrees/s. Processing fps is laptop throughput with CSV and video writing; it is not Pi performance.

For comparison, the largest changes immediately before the final low-pass/rate limiter were:

| Recording | Upstream offset step (lane widths) | Upstream steering step (degrees) |
| --- | ---: | ---: |
| near_hostel.mp4 | 8.509 | 135.993 |
| nmv_road.mp4 | 1.135 | 69.048 |
| perfect_roada.mp4 | 0.943 | 129.367 |
| road_video.mp4 | 0.396 | 12.022 |

These upstream values are measured inside the updated pipeline after its geometry/adaptive filters. They are **not** a separate benchmark of the previous repository version. Large upstream excursions show why the final bounds matter, and also why perception quality still needs validation.

## Artifacts and limits

Artifacts are in `outputs/regression/`: four `*_lane.mp4` videos, four `*_steering.csv` logs, per-video summaries, `report.json`, `config_used.json` and `unit-tests.txt`. The outputs directory is intentionally ignored by Git.

Fresh-lane percentage excludes held and lost estimates; it is not detection accuracy. Control also disables on obstacle indications. In `near_hostel`, only 53.4% of frames had fresh accepted lanes. Sample frames show false obstacle indications on otherwise clear road. These are material perception limitations: passing smoothing checks does not establish autonomous driving reliability. There are no ground-truth annotations, motor/linkage measurements or closed-loop vehicle tests. Smoothing introduces lag, especially after a large direction change.

Pi capture, Hailo and serial protocol logic were tested with mocks. No physical Pi, camera, encoder, motor or compatible HEF was available. Those deployment checks, sensor/driver-specific bridge firmware, calibration and physical watchdog enforcement remain outstanding; see [Pi deployment](pi.md) and [motor integration](hardware.md).
