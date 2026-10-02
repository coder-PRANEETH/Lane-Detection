# Raspberry Pi 5 deployment

The project supports USB cameras through OpenCV, Raspberry Pi camera modules through Picamera2, and a separate Hailo semantic segmentation backend. The camera and Hailo adapters have automated tests using simulated devices. **A physical Pi, camera, Hailo device and steering motor were not available for hardware validation.** There is no measured Pi frame rate or validated Hailo model included.

## Install the Pi software

Use 64-bit Raspberry Pi OS. The current [official AI setup guide](https://www.raspberrypi.com/documentation/computers/ai.html) targets Trixie and covers hardware installation, OS updates and the matching accelerator packages. Identify the accelerator before installing its runtime:

| Hardware | Accelerator | Raspberry Pi package |
| --- | --- | --- |
| AI Kit / AI HAT+ | Hailo-8L or Hailo-8 | `hailo-all` |
| AI HAT+ 2 | Hailo-10H | `hailo-h10-all` |

These package families cannot coexist. This project's synchronous `InferVStreams` adapter targets the Hailo-8/8L HailoRT API. Hailo-10 compatibility is **not established**; installing its package alone does not demonstrate that this adapter or a Hailo-8 HEF will work.

For Hailo-8/8L, after completing the OS setup in the official guide:

```bash
sudo apt update
sudo apt install dkms hailo-all
sudo apt install python3-venv python3-opencv python3-numpy python3-picamera2 python3-serial rpicam-apps
sudo reboot
```

After reboot, check the accelerator with `hailortcli fw-control identify`. Keep the driver, firmware, runtime and model toolchain compatible; the [official package guidance](https://www.raspberrypi.com/documentation/computers/ai.html#package-versions-for-ai-kit-and-ai-hat) explains version matching.

From the project directory, create a separate Pi environment that can see the system camera and Hailo bindings:

```bash
python3 -m venv --system-site-packages .venv-pi
source .venv-pi/bin/activate
python -c "import cv2, numpy, picamera2, serial; from hailo_platform import HEF, InferVStreams"
python -m unittest discover -s tests
```

`--system-site-packages` makes the distribution's native camera bindings available; see the [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf). If the Hailo import fails, finish installing the matching HailoRT Python bindings through the Pi/Hailo distribution. This backend does not import or require Torch. The root `requirements.txt` also installs the laptop Torch backend, so it is unnecessary for this Pi environment.

## Supply a compatible road model

The existing `models/yolopv2.pt` is a TorchScript model. Hailo requires a separately compiled `.hef` targeting the actual accelerator and compatible runtime. No compatible road HEF is supplied or automatically downloaded. Obtaining or compiling and validating that model remains a deployment requirement.

The adapter expects:

- Exactly one network group and one image input, with HWC metadata and three channels. It supplies **RGB uint8 pixels, 0–255**, using aspect-preserving resizing and padding of 114. The HEF must include whatever normalization the model requires.
- A dense semantic segmentation output whose HEF metadata is `(height, width, channels)`. Choose its exact name when the model has several outputs. An object detector or an instance segmentation output requiring separate decoding does not meet this interface.
- `output_layout`: `HWC` or `CHW`. This configures HailoRT's host output order and is checked against the returned tensor.
- `output_kind`: `logits`, `probabilities`, or `class_ids`. Logits use softmax for multiple channels and sigmoid for a single binary channel. Probabilities must be in `[0, 1]`; class IDs must be nonnegative integers in one channel.
- The model's actual `road_class`. It defaults to `1`; a different training label map needs a different value. For a single binary probability/logit channel, class `1` means the positive class and class `0` its complement.

Input size comes from the HEF; `model.input_width` applies only to Torch. The adapter removes letterbox padding and returns a road mask at the camera frame size. It fails with an error on incompatible tensors or missing hardware instead of changing backends. See the [implementation](../road/hailo_segmenter.py) and [HailoRT VStreams API](https://github.com/hailo-ai/hailort/blob/master/hailort/libhailort/bindings/python/platform/hailo_platform/pyhailort/pyhailort.py).

Inspect the HEF's names and dimensions before configuring it:

```bash
python -c "from hailo_platform import HEF; m=HEF('models/road_segmentation.hef'); print(m.get_input_vstream_infos()); print(m.get_output_vstream_infos())"
```

## Choose a camera or recording

For a Raspberry Pi camera module, verify the camera first using `rpicam-hello --list-cameras` and `rpicam-hello -n -t 5000`. These utilities use the current Raspberry Pi camera stack; see the [official camera guide](https://www.raspberrypi.com/documentation/computers/camera_software.html). Use `source.mode: "picamera2"` for that camera. The adapter requests `RGB888`, which Picamera2 exposes as BGR bytes for OpenCV, and requests new frames with `queue=False`.

Save the following as `config.pi.json` **in the project root**, replacing the HEF filename and output settings with those of your model. Omitted keys inherit the defaults from [road/config.py](../road/config.py). Paths are relative to the configuration file.

```json
{
  "source": {
    "mode": "picamera2",
    "device": 0,
    "width": 640,
    "height": 360,
    "fps": 30,
    "read_timeout": 2
  },
  "model": {
    "backend": "hailo",
    "hailo": {
      "hef": "models/road_segmentation.hef",
      "input_name": null,
      "output_name": null,
      "output_layout": "HWC",
      "output_kind": "logits",
      "road_class": 1
    }
  },
  "steering": {"mode": "simulation", "enabled": false},
  "output": {"show": false, "save_video": false, "csv": true}
}
```

Use `null` names only when that input/output is unambiguous. For a USB camera, change `mode` to `"camera"`; `device` can be `0` or `"/dev/video0"`. For a recording, use `mode: "video"` and `path: "videos_lowres/road_video.mp4"`. Both live modes keep only the latest available frame, so slower inference does not accumulate an application queue. Requested USB camera resolution and frame rate depend on the device's supported modes.

```bash
python run.py --config config.pi.json --max-frames 120
python run.py --config config.pi.json
```

The first command processes a short capture; the second runs continuously until Ctrl-C. CSV logs include capture age, lane measurements, target angle, measured angle and controller status. In this configuration, the measured angle is simulated and physical motor output is disabled.

Use the measured `frame_age_s` and processing rate to assess the Pi installation. The default controller rejects frames older than 0.3 seconds. Reducing camera resolution alone does not reduce a fixed-size HEF's inference workload. Model choice and actual inference timing must be validated before configuring motor feedback; see the [project instructions](../README.md) for calibration, the serial interface and stop behavior.
