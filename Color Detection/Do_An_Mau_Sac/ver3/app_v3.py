# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
import time

# ==========================================
# 1. CẤU HÌNH (Dũng chỉnh ở đây)
# ==========================================
# Dũng có thể để "color.onnx" hoặc "color.pt" đều được
MODEL_PATH = "best.onnx"

# Thứ tự màu Dũng đã xác nhận: Blue = 0, Red = 1, Yellow = 2
CLASS_NAMES = ["Blue", "Red", "Yellow"]

# Ngưỡng tin cậy (chỉnh từ 0.3 - 0.6)
CONF_THRESHOLD = 0.5

# ==========================================
# 2. KHỞI TẠO HỆ THỐNG
# ==========================================
print(f"[INFO] Đang nạp mô hình bằng Ultralytics: {MODEL_PATH}")
# Thư viện này tự động xử lý các lớp DFL/Reshape cho Dũng
model = YOLO(MODEL_PATH)

# Mở Webcam
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Hệ thống sẵn sàng! Nhấn 'q' để thoát.")

# ==========================================
# 3. VÒNG LẶP XỬ LÝ CHÍNH
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break

    start_t = time.time()

    # CHẠY NHẬN DIỆN (Sử dụng stream=True để tiết kiệm bộ nhớ)
    # verbose=False để terminal không bị nhảy chữ liên tục
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 1. Lấy tọa độ pixel (x1, y1, x2, y2)
            b = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

            # 2. Lấy ID lớp và Độ tin cậy
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 3. Vẽ Bounding Box
            # Màu xanh lá (0, 255, 0) cho dễ nhìn
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 4. Hiển thị nhãn
            label_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"ID:{cls_id}"
            label_text = f"{label_name} {conf * 100:.1f}%"

            # Vẽ nền cho chữ để dễ đọc
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tính FPS
    fps = 1 / (time.time() - start_t)
    cv2.putText(frame, f"Engine: Ultralytics | FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Hiển thị kết quả
    cv2.imshow("DO AN TOT NGHIEP - DUNG ROBOT", frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
print("[INFO] Đã dừng hệ thống.")