# -*- coding: utf-8 -*-
import cv2
import time
from picamera2 import Picamera2

# 1. KHỞI TẠO CAMERA
print("[INFO] Đang mở camera để đo FPS thuần...")
picam2 = Picamera2()
# Cấu hình 640x480 là mức cân bằng nhất cho Pi 4
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Biến để tính FPS trung bình
frame_count = 0
start_time = time.time()

try:
    while True:
        # Lấy 1 frame từ camera
        frame = picam2.capture_array()
        if frame is None:
            continue

        frame_count += 1

        # Tính toán FPS tức thời
        now = time.time()
        elapsed = now - start_time

        # Cập nhật FPS mỗi giây một lần để tránh số nhảy quá nhanh
        if elapsed > 1.0:
            fps = frame_count / elapsed
            print(f"Current FPS: {fps:.2f}")
            # Reset để tính lượt tiếp theo
            frame_count = 0
            start_time = now

        # HIỂN THỊ: Chuyển RGB sang BGR để không bị xanh lè
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Vẽ FPS lên màn hình (nếu muốn xem trực tiếp)
        cv2.putText(display_frame, f"RAW FPS: {frame_count if elapsed < 1 else fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camera Raw FPS Test", display_frame)

        # Nhấn 'q' để dừng
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    print("[INFO] Đang đóng camera...")
    picam2.stop()
    cv2.destroyAllWindows()