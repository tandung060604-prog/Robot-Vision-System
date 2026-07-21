import cv2
from ultralytics import YOLO

# 1. Load mô hình đã huấn luyện của bạn
# Đảm bảo file best.pt nằm cùng thư mục với file code này
model = YOLO("best.pt")

# 2. Mở Camera (0 là webcam mặc định của laptop)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

print("Đang chạy nhận diện... Nhấn 'q' để thoát.")

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Chạy mô hình dự đoán trên khung hình
    # conf=0.6: Chỉ hiển thị vật thể có độ tin cậy trên 60% (dựa trên biểu đồ F1 của bạn)
    results = model(frame, conf=0.65, iou=0.45, stream=True)

    for r in results:
        # Vẽ kết quả lên khung hình
        annotated_frame = r.plot()

        # Hiển thị khung hình đã vẽ
        cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    # Nhấn 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()