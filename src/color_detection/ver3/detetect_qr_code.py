import cv2
import numpy as np


# --- BƯỚC 1: NẠP DỮ LIỆU CHUẨN ---
def lay_du_lieu_mau(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Lỗi: Không tìm thấy file {path}")
        return None
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img)
    return data


# Tạo từ điển đối soát
ma_chuan = {
    lay_du_lieu_mau("anh_mau_1.png"): "MAU 1",
    lay_du_lieu_mau("anh_mau_2.png"): "MAU 2",
    lay_du_lieu_mau("anh_mau_3.png"): "MAU 3"
}

print("--- HỆ THỐNG SẴN SÀNG QUÉT TỔNG HỢP ---")

# --- BƯỚC 2: QUÉT CAMERA ---
cap = cv2.VideoCapture(0)  # Mở camera tại điểm B [cite: 9, 16]
detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    if not ret: break

    # Sử dụng detectAndDecodeMulti để quét tất cả mã trong hình
    ret_qr, decoded_info, points, _ = detector.detectAndDecodeMulti(frame)

    if ret_qr:
        danh_sach_hien_tai = []  # Danh sách lưu các mã thấy trong khung hình này

        for i, data in enumerate(decoded_info):
            if data in ma_chuan:
                ten_mau = ma_chuan[data]
                danh_sach_hien_tai.append(ten_mau)

                # Vẽ khung và tên cho từng mã QR
                if points is not None:
                    pts = points[i].astype(int)
                    for j in range(len(pts)):
                        cv2.line(frame, tuple(pts[j]), tuple(pts[(j + 1) % 4]), (0, 255, 0), 3)

                    cv2.putText(frame, ten_mau, (pts[0][0], pts[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Báo cáo tổng hợp ra Terminal (Input/Output area)
        if danh_sach_hien_tai:
            print(f"Trong hinh dang co: {', '.join(danh_sach_hien_tai)}")

    cv2.imshow("He thong quet da ma QR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()