# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import tflite_runtime.interpreter as tflite

# ==========================================
# 1. CẤU HÌNH
# ==========================================
MODEL_PATH = 'drug_int8.tflite'
CONF_THRESHOLD = 0.35 
NMS_THRESHOLD = 0.45
CLASSES = ['Carbo', 'Vaseline', 'Vitamin C']

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
        
        # FIX: Chuyển BGR sang RGB để AI nhận diện đúng Class
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        if is_int8:
            input_data = np.expand_dims(img_rgb, axis=0).astype(input_details[0]['dtype'])
        else:
            input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

        # --- BƯỚC 2: INFERENCE ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # --- BƯỚC 3: HẬU XỬ LÝ ---
        # Chuyển về shape [8400, 7]
        output = output[0].transpose()

        # Nếu là model INT8, cần giải mã giá trị (Dequantize) để lấy xác suất thực
        # Tuy nhiên, np.argmax vẫn có thể chạy trên số nguyên nếu scale đồng nhất.
        # Để an toàn, chúng ta lấy class_scores trực tiếp.

        boxes, confidences, class_ids = [], [], []

        for pred in output:
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            conf = class_scores[class_id]

            # Với INT8, giá trị conf có thể rất lớn nếu chưa dequantize, 
            # nhưng tỷ lệ giữa các class vẫn giữ nguyên.
            if conf > (CONF_THRESHOLD * 255 if is_int8 else CONF_THRESHOLD):
                cx, cy, bw, bh = pred[0:4]

                # Tỷ lệ scale chính xác (YOLOv8 TFLite thường trả về tọa độ pixel dựa trên input_size)
                # Nếu model trả về tọa độ 0~1:
                x = int((cx - bw / 2) * w_orig / input_size) if cx > 1.1 else int((cx - bw / 2) * w_orig)
                y = int((cy - bh / 2) * h_orig / input_size) if cy > 1.1 else int((cy - bh / 2) * h_orig)
                width = int(bw * w_orig / input_size) if bw > 1.1 else int(bw * w_orig)
                height = int(bh * h_orig / input_size) if bh > 1.1 else int(bh * h_orig)

                boxes.append([x, y, width, height])
                confidences.append(float(conf))
                class_ids.append(class_id)

        # NMS lọc trùng
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{CLASSES[class_ids[i]]}" # Hiển thị tên class

                # Vẽ
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