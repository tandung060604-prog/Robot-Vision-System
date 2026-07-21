# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
MODEL_PATH = 'drug_int8.tflite'
CONF_THRESHOLD = 0.3  # Hạ xuống 0.3 để dễ test lúc đầu
NMS_THRESHOLD = 0.45  # Ngưỡng lọc các khung hình đè lên nhau
CLASSES = ['Vitamin C', 'Carbo', 'Vaseline']

# Khởi tạo Interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_size = input_shape[1]  # Thường là 640
is_int8 = input_details[0]['dtype'] == np.int8 or input_details[0]['dtype'] == np.uint8

print(f"[INFO] Model Type: {'INT8' if is_int8 else 'FLOAT32'}")
print(f"[INFO] Input Size: {input_size}x{input_size}")

# ==========================================
# 2. KHỞI TẠO PICAMERA2
# ==========================================
picam2 = Picamera2()
# Cấu hình 640x480 để cân bằng giữa tốc độ và độ chi tiết
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.start()

print("[INFO] He thong san sang! Dang quet camera...")

try:
    while True:
        raw_frame = picam2.capture_array()
        if raw_frame is None: continue

        h_orig, w_orig = raw_frame.shape[:2]
        start_t = time.time()

        # --- BƯỚC 1: TIỀN XỬ LÝ (PRE-PROCESS) ---
        img = cv2.resize(raw_frame, (input_size, input_size))
        if is_int8:
            # Model INT8 yêu cầu dữ liệu 0-255 kiểu uint8/int8
            input_data = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])
        else:
            # Model FLOAT32 yêu cầu chuẩn hóa 0.0 - 1.0
            input_data = (img.astype(np.float32) / 255.0)
            input_data = np.expand_dims(input_data, axis=0)

        # --- BƯỚC 2: CHẠY MODEL (INFERENCE) ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # --- BƯỚC 3: HẬU XỬ LÝ (POST-PROCESS) ---
        # YOLOv8 TFLite output thường là [1, 7, 8400] -> chuyển về [8400, 7]
        if output.shape[1] < output.shape[2]:
            output = output[0].transpose()
        else:
            output = output[0]

        boxes = []
        confidences = []
        class_ids = []

        for pred in output:
            # pred[:4] là x, y, w, h | pred[4:] là xác suất của các class
            probs = pred[4:]
            class_id = np.argmax(probs)
            conf = probs[class_id]

            if conf > CONF_THRESHOLD:
                # YOLOv8 trả về tâm x, tâm y, chiều rộng, chiều cao
                cx, cy, w, h = pred[:4]

                # Chuyển đổi tọa độ về pixel thực tế trên màn hình 640x480
                # Nếu giá trị < 1.1 thì là tọa độ chuẩn hóa, cần nhân với kích thước ảnh
                scale_x = w_orig if cx <= 1.1 else w_orig / input_size
                scale_y = h_orig if cy <= 1.1 else h_orig / input_size

                x = int((cx - w / 2) * scale_x)
                y = int((cy - h / 2) * scale_y)
                width = int(w * scale_x)
                height = int(h * scale_y)

                boxes.append([x, y, width, height])
                confidences.append(float(conf))
                class_ids.append(class_id)

        # Áp dụng NMS để lọc bớt khung hình đè lên nhau
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

        # --- BƯỚC 4: VẼ KẾT QUẢ ---
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{CLASSES[class_ids[i]]}: {confidences[i]:.2f}"

                # Vẽ khung
                cv2.rectangle(raw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Vẽ nhãn
                cv2.putText(raw_frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # In tọa độ tâm để Robot gắp
                print(f"PICK: {CLASSES[class_ids[i]]} at ({x + w / 2:.0f}, {y + h / 2:.0f})")

        # Tính toán và hiển thị FPS
        fps = 1.0 / (time.time() - start_t)
        cv2.putText(raw_frame, f"FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Robot Vision", raw_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("[INFO] Dang dong camera...")
    picam2.stop()
    cv2.destroyAllWindows()