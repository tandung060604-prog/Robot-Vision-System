# Robot Vision System

> Real-time, edge-ready visual perception for a mobile robot.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**Robot Vision System** is a graduation project that equips a mobile robot with real-time visual perception. A camera stream is processed by an optimized TensorFlow Lite model, then the system reports detected objects, their locations, confidence, and live FPS. The design targets deployment on resource-constrained robot computers while remaining easy to run on a laptop during development.

### Key capabilities

- **Real-time detection** from a USB/web camera with visual bounding boxes.
- **Edge inference** using TensorFlow Lite (`float16`, `float32`, or quantized `int8` models).
- **Robot-relevant classes**: medicine boxes (`hop_thuoc`), colored blocks (`khoi_mau`), and beverage cans (`nuoc_ngot`).
- **Tunable detection pipeline**: confidence threshold, NMS IoU threshold, camera index, resolution, and model path are configurable from the command line.
- **Diagnostics**: clear startup validation, frame-rate overlay, and structured console events that can be connected to robot control logic.

## Project structure

```text
.
├── docs/                         # Reports, component list, and project documentation
├── python/
│   └── object_detection/
│       ├── test_detect.py         # Configurable real-time TFLite inference application
│       ├── detect.py              # YOLO/OpenCV inference prototype
│       ├── best_int8.tflite       # Quantized edge model
│       ├── metadata.yaml          # Model metadata
│       └── sample_image.jpg       # Example input image
├── requirements.txt               # Reproducible Python dependencies
├── .gitignore
└── LICENSE
```

## Quick start

### 1. Clone and create an environment

```bash
git clone https://github.com/tandung060604-prog/Robot-Vision-System.git
cd Robot-Vision-System
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run real-time detection

```bash
cd python/object_detection
python test_detect.py --model best_int8.tflite --camera 0 --confidence 0.61
```

Press `q` in the preview window to exit. If the robot camera is attached as another device, try `--camera 1`.

## Configuration

```bash
python test_detect.py --help
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | `best_int8.tflite` | Path to a TensorFlow Lite object-detection model |
| `--camera` | `0` | OpenCV camera device index |
| `--confidence` | `0.61` | Minimum detection confidence (0–1) |
| `--nms-iou` | `0.45` | Overlap threshold used for non-maximum suppression |
| `--width`, `--height` | `640`, `480` | Requested camera resolution |

## Model notes

The TFLite models are included for convenient demonstration. The application automatically handles normalized floating-point input and quantized integer input, including the model's quantization scale and zero point. For a custom model, keep the expected YOLO-style output layout or adapt `decode_predictions()` in `test_detect.py`.

## Documentation

Project reports and the hardware component list are available in [`docs/`](docs/). They provide the research context, robot architecture, and implementation details behind this prototype.

## Roadmap

- [ ] Publish a short hardware/demo video and inference benchmark.
- [ ] Connect detected target coordinates to the robot navigation/manipulator controller.
- [ ] Add automated tests using recorded camera frames.
- [ ] Export trained models and datasets through versioned releases.

## License

Distributed under the [MIT License](LICENSE).
