# Mobile Robot Vision System

Dự án Đồ án Tốt nghiệp: Hệ thống nhận diện hình ảnh cho Mobile Robot (Nhận diện màu sắc, Nhận diện thuốc, Nhận diện nước tăng lực, Nhận diện vật thể).

## 🚀 Giới thiệu
Hệ thống sử dụng các mô hình học sâu (Deep Learning) để thực hiện các bài toán nhận diện đối tượng theo thời gian thực (Real-time Object Detection) thông qua camera gắn trên robot. Các module bao gồm:
- **Color Detection**: Nhận diện màu sắc và phân loại tín hiệu.
- **Drug Detection**: Nhận diện các loại thuốc.
- **Energy Cans Detection**: Nhận diện lon nước tăng lực.
- **Object Detection**: Nhận diện vật thể nói chung.

## 📁 Cấu trúc thư mục

```text
/
├── src/                      # Source code chính của đồ án
│   ├── color_detection/      # Code nhận diện màu sắc
│   ├── drug_detection/       # Code nhận diện thuốc
│   ├── energy_cans_detection/# Code nhận diện lon nước tăng lực
│   └── object_detection/     # Code nhận diện vật thể chung
├── docs/                     # Tài liệu đồ án, báo cáo
├── data/                     # Data test, dataset, media (đã bị bỏ qua trên git)
├── .gitignore                # File cấu hình Git ignore
└── README.md                 # Tài liệu hướng dẫn
```

## ⚙️ Cài đặt & Hướng dẫn sử dụng

### 1. Yêu cầu hệ thống
- Python 3.8+
- Các thư viện cần thiết: `opencv-python`, `torch`, `torchvision`, `ultralytics`, `flask` (nếu chạy web server), v.v.

### 2. Cài đặt thư viện
Bạn có thể cài đặt các thư viện cần thiết cho từng module bằng cách chạy lệnh:
```bash
pip install -r requirements.txt # (Nếu có file requirements)
```
Hoặc cài đặt thủ công dựa trên các file code trong thư mục `src/`.

### 3. Chạy chương trình
Di chuyển vào từng module tương ứng trong `src/` và chạy file Python chính (ví dụ `main.py`, `app_v2.py`, v.v.).

```bash
cd src/color_detection
python app_v2.py
```

## 📝 Lưu ý
- Các file mô hình có kích thước lớn (như `.pt`, `.pb`) hoặc các video demo, dataset nằm trong thư mục `data/` đã được thiết lập bỏ qua trong file `.gitignore` để tránh phình to dung lượng kho lưu trữ trên GitHub. 
- Vui lòng chỉ đưa lên các mô hình tối ưu đã nén như `.tflite` (dưới 100MB).

## 📜 Giấy phép
Dự án được phân phối dưới giấy phép MIT. Xem thêm tại file `LICENSE`.
