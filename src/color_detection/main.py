import os
import cv2
from ultralytics import YOLO

# --- CẤU HÌNH HỆ THỐNG ---
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Sửa lỗi DLL Windows

# 1. Tên file mô hình .pt của bạn
MODEL_NAME = 'color.pt'

# 2. NGƯỠNG TIN CẬY (CONFIDENCE THRESHOLD) - CHÌA KHÓA CỦA BẠN
# Tăng từ 0.5 lên 0.8. Nghĩa là AI phải chắc chắn > 80% mới hiển thị.
# Bạn có thể tăng lên 0.85 hoặc 0.9 nếu vẫn bị nhầm.
CONFIDENCE_THRESHOLD = 0.8

# 3. NGƯỠNG GIAO NHAU (IOU THRESHOLD) - GIẢM DUPLICATE KHUNG BÁO
# Dùng cho NMS (Non-Maximum Suppression). Thấp hơn làm NMS nghiêm ngặt hơn.
IOU_THRESHOLD = 0.45 # Mặc định thường là 0.7

# --- KHỞI TẠO MÔ HÌNH VÀ CAMERA ---
print(f"--- Đang nạp mô hình {MODEL_NAME} ---")
# Load mô hình và di chuyển vào CPU (hoặc GPU nếu có)
model = YOLO(MODEL_NAME).to('cpu')

print("--- Đang bật Camera ---")
# Dùng CAP_DSHOW cho Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Lỗi: Không thể mở camera.")
    exit()

# Thiết lập độ phân giải chuẩn (giảm tải cho CPU)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("--- BẮT ĐẦU DETECT. Nhấn 'q' để thoát ---")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không đọc được khung hình.")
        break

    # --- NHẬN DIỆN VỚI THÔNG SỐ TINH CHỈNH ---
    # stream=True: tối ưu cho video real-time
    # conf: Ngưỡng tin cậy (conf) -> lọc vật thể lạ
    # iou: Ngưỡng NMS (iou) -> lọc khung bao trùng nhau
    results = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        stream=True
    )

    for r in results:
        # Sử dụng hàm vẽ plot() của Ultralytics
        # (Hiển thị nhãn, confidence ngay trên bounding box)
        annotated_frame = r.plot()

        # Hiển thị
        cv2.imshow("Dung AI Robot - Anti False Positives", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
print("--- Đã tắt hệ thống ---")