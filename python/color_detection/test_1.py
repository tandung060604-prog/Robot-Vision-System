# -*- coding: utf-8 -*-
import cv2
import time
from picamera2 import Picamera2

# KHỞI TẠO WECHAT QR
detector = cv2.wechat_qrcode_WeChatQRCode(
    "detect.prototxt", "detect.caffemodel",
    "sr.prototxt", "sr.caffemodel"
)

# KHỞI TẠO CAMERA (BGR chuẩn OpenCV)
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.start()

print("[INFO] Đưa mã QR ra trước camera để kiểm tra độ nhạy...")

try:
    while True:
        frame = picam2.capture_array()
        if frame is None: continue

        # Nhận diện
        res, points = detector.detectAndDecode(frame)

        if len(res) > 0:
            for i in range(len(res)):
                content = res[i]
                pts = points[i].astype(int)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                cv2.putText(frame, content, (pts[0][0], pts[0][1] - 10), 1, 1.5, (0, 255, 0), 2)
                print(f"QR Content: {content}")

        cv2.imshow("TEST WECHAT QR - DUNG ROBOT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    picam2.stop()
    cv2.destroyAllWindows()