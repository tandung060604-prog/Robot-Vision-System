#!/usr/bin/env python3
"""Run real-time object detection with a TensorFlow Lite model and a camera."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:  # TensorFlow provides a compatible interpreter on Windows.
    try:
        from tensorflow import lite as tflite
    except ImportError as error:
        raise SystemExit(
            "TensorFlow Lite runtime not found. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error


CLASSES = ("hop_thuoc", "khoi_mau", "nuoc_ngot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="best_int8.tflite", help="Path to a .tflite model")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera device index")
    parser.add_argument("--confidence", type=float, default=0.61, help="Minimum score (0-1)")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold (0-1)")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height")
    return parser.parse_args()


def dequantize(tensor: np.ndarray, details: dict) -> np.ndarray:
    """Return a float tensor using the TFLite quantization parameters, if present."""
    scale, zero_point = details["quantization"]
    if scale:
        return (tensor.astype(np.float32) - zero_point) * scale
    return tensor.astype(np.float32)


def prepare_input(frame: np.ndarray, input_details: dict, input_size: int) -> np.ndarray:
    image = cv2.cvtColor(cv2.resize(frame, (input_size, input_size)), cv2.COLOR_BGR2RGB)
    dtype = input_details["dtype"]
    if np.issubdtype(dtype, np.floating):
        return np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

    scale, zero_point = input_details["quantization"]
    if not scale:
        return np.expand_dims(image.astype(dtype), axis=0)
    quantized = np.round(image / 255.0 / scale + zero_point)
    limits = np.iinfo(dtype)
    return np.expand_dims(np.clip(quantized, limits.min, limits.max).astype(dtype), axis=0)


def decode_predictions(output: np.ndarray, input_size: int, frame_width: int, frame_height: int, confidence: float):
    """Decode a YOLO-style [x, y, width, height, class scores...] tensor."""
    predictions = output.squeeze()
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected model output shape: {output.shape}")
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes, scores, class_ids = [], [], []
    for prediction in predictions:
        if len(prediction) < 5:
            continue
        class_id = int(np.argmax(prediction[4:]))
        score = float(prediction[4 + class_id])
        if score < confidence:
            continue
        center_x, center_y, width, height = prediction[:4]
        if max(center_x, center_y, width, height) > 1.5:
            center_x, width = center_x * frame_width / input_size, width * frame_width / input_size
            center_y, height = center_y * frame_height / input_size, height * frame_height / input_size
        else:
            center_x, width = center_x * frame_width, width * frame_width
            center_y, height = center_y * frame_height, height * frame_height
        boxes.append([int(center_x - width / 2), int(center_y - height / 2), int(width), int(height)])
        scores.append(score)
        class_ids.append(class_id)
    return boxes, scores, class_ids


def main() -> None:
    args = parse_args()
    if not 0 <= args.confidence <= 1 or not 0 <= args.nms_iou <= 1:
        raise SystemExit("--confidence and --nms-iou must be between 0 and 1.")
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path.resolve()}")

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_size = int(input_details["shape"][1])

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}.")

    print(f"[INFO] Model: {model_path.name} | Input: {input_size}x{input_size} | Camera: {args.camera}")
    print("[INFO] Press 'q' to exit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[WARN] Failed to read camera frame.", file=sys.stderr)
                break
            start = time.perf_counter()
            interpreter.set_tensor(input_details["index"], prepare_input(frame, input_details, input_size))
            interpreter.invoke()
            output = dequantize(interpreter.get_tensor(output_details["index"]), output_details)
            boxes, scores, class_ids = decode_predictions(output, input_size, frame.shape[1], frame.shape[0], args.confidence)
            indices = cv2.dnn.NMSBoxes(boxes, scores, args.confidence, args.nms_iou) if boxes else []
            for index in np.array(indices).flatten():
                x, y, width, height = boxes[index]
                name = CLASSES[class_ids[index]] if class_ids[index] < len(CLASSES) else f"class_{class_ids[index]}"
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 70), 2)
                cv2.putText(frame, f"{name} {scores[index]:.0%}", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 70), 2)
                print(f"DETECT: {name} ({scores[index]:.0%}) at ({x + width // 2}, {y + height // 2})")
            fps = 1.0 / max(time.perf_counter() - start, 1e-6)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 230), 2)
            cv2.imshow("Robot Vision System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
