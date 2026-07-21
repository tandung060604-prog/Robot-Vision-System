import cv2
from ultralytics import YOLO

# 1. Load model đã train của bạn
# Đảm bảo file 'best.pt' nằm cùng thư mục với file code này
model = YOLO('best.pt')

# 2. Mở Webcam (0 thường là webcam mặc định của laptop)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

print("Đang chạy model... Nhấn 'q' để thoát.")

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Chạy inference (nhận diện)
    # conf=0.5: Chỉ hiển thị vật thể nếu độ tự tin trên 50%
    results = model(frame, stream=True, conf=0.5)

    # 4. Hiển thị kết quả lên màn hình
    for r in results:
        annotated_frame = r.plot()  # Vẽ bounding box và label lên frame

        # Hiển thị frame đã được vẽ
        cv2.imshow("YOLOv8 Drug Detection - Test", annotated_frame)

    # Nhấn phím 'q' để dừng
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()