# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import tflite_runtime.interpreter as tflite

# ==========================================
# 1. CẤU HÌNH ĐỐI TƯỢNG MỚI
# ==========================================
# Đảm bảo file tflite của 3 lon nước đã nằm trong thư mục
MODEL_PATH = 'energy_int8.tflite'
# Nâng threshold lên để tránh bắt nhầm cái ghế/cái đầu
CONF_THRESHOLD = 0.65
NMS_THRESHOLD = 0.45
# Cập nhật đúng tên 3 loại lon nước của Dũng
CLASSES = ['birdy', 'bo_huc', 'boss_coffee']

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]
is_int8 = input_details[0]['dtype'] in (np.int8, np.uint8)

print(f"[INFO] Đang chạy nhận diện: {CLASSES}")
print(f"[INFO] Mode: {'INT8' if is_int8 else 'FLOAT32'} | Thresh: {CONF_THRESHOLD}")

# ==========================================
# 2. KHỞI TẠO CAMERA
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

        # --- BƯỚC 1: TIỀN XỬ LÝ (CHỈNH MÀU RGB) ---
        img_resized = cv2.resize(frame, (input_size, input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # YOLO cần RGB

        if is_int8:
            input_data = np.expand_dims(img_rgb, axis=0).astype(input_details[0]['dtype'])
        else:
            input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

        # --- BƯỚC 2: INFERENCE (CHẠY AI) ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # --- BƯỚC 3: HẬU XỬ LÝ & LỌC NHIỄU ---
        output = output[0].transpose()  # Shape [8400, 7]

        boxes, confidences, class_ids = [], [], []

        for pred in output:
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            conf = class_scores[class_id]

            # Xử lý ngưỡng tin cậy cho INT8
            thresh = CONF_THRESHOLD * 255 if is_int8 else CONF_THRESHOLD

            if conf > thresh:
                cx, cy, bw, bh = pred[0:4]

                # Chuyển tọa độ về kích thước ảnh gốc
                x = int((cx - bw / 2) * w_orig / input_size)
                y = int((cy - bh / 2) * h_orig / input_size)
                width = int(bw * w_orig / input_size)
                height = int(bh * h_orig / input_size)

                boxes.append([x, y, width, height])
                confidences.append(float(conf))
                class_ids.append(class_id)

        # NMS lọc các khung hình chồng lấn
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

            # Kiểm tra nếu indices không trống
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    label = f"{CLASSES[class_ids[i]]}"

                    # TÍNH TỌA ĐỘ TÂM (Phục vụ gắp vật)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Vẽ Bounding Box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    print(f"PHÁT HIỆN: {label:12} | Tâm: ({center_x}, {center_y})")

        # Hiển thị FPS thực tế trên Pi
        fps = 1.0 / (time.time() - start_t)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Robot Vision - Soda Can Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()