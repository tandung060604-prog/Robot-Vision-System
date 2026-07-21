# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import tflite_runtime.interpreter as tflite

# ==========================================
# 1. CẤU HÌNH (ĐÃ CẬP NHẬT CHO ĐỒ ÁN)
# ==========================================
MODEL_PATH = 'best_int8.tflite'
CONF_THRESHOLD = 0.61           # Ngưỡng tối ưu từ biểu đồ F1-Curve
NMS_THRESHOLD = 0.45            # Ngưỡng lọc trùng khung hình
CLASSES = ['hop_thuoc', 'khoi_mau', 'nuoc_ngot']

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]
is_int8 = input_details[0]['dtype'] in (np.int8, np.uint8)

print(f"[INFO] Model: {'INT8' if is_int8 else 'FLOAT32'} | Input: {input_size}x{input_size}")

# ==========================================
# 2. WEBCAM
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue

        h_orig, w_orig = frame.shape[:2]
        start_t = time.time()

        # --- BƯỚC 1: TIỀN XỬ LÝ (SỬA LỖI MÀU) ---
        # Resize ảnh
        img_resized = cv2.resize(frame, (input_size, input_size))

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        if is_int8:
            input_data = np.expand_dims(img_rgb, axis=0).astype(input_details[0]['dtype'])
        else:
            input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

        # --- BƯỚC 2: INFERENCE ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        output = output[0].transpose()

        boxes, confidences, class_ids = [], [], []

        for pred in output:
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            conf = class_scores[class_id]

            if conf > (CONF_THRESHOLD * 255 if is_int8 else CONF_THRESHOLD):
                cx, cy, bw, bh = pred[0:4]

                x = int((cx - bw / 2) * w_orig / input_size) if cx > 1.1 else int((cx - bw / 2) * w_orig)
                y = int((cy - bh / 2) * h_orig / input_size) if cy > 1.1 else int((cy - bh / 2) * h_orig)
                width = int(bw * w_orig / input_size) if bw > 1.1 else int(bw * w_orig)
                height = int(bh * h_orig / input_size) if bh > 1.1 else int(bh * h_orig)

                boxes.append([x, y, width, height])
                confidences.append(float(conf))
                class_ids.append(class_id)

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{CLASSES[class_ids[i]]}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                print(f"DETECT: {label} at center ({x+w//2}, {y+h//2})")

        # Hiển thị FPS
        fps = 1.0 / (time.time() - start_t)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Robot Vision Fixed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()