import urllib.request
import os

# Link dự phòng từ kho lưu trữ chính thức của WeChat CV
urls = {
    "detect.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_wechat_qrcode_models/master/detect.prototxt",
    "detect.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_wechat_qrcode_models/master/detect.caffemodel",
    "sr.prototxt": "https://raw.githubusercontent.com/WeChatCV/opencv_wechat_qrcode_models/master/sr.prototxt",
    "sr.caffemodel": "https://raw.githubusercontent.com/WeChatCV/opencv_wechat_qrcode_models/master/sr.caffemodel"
}

print("--- Đang tải bộ não WeChat QR (vui lòng đợi)... ---")

for name, url in urls.items():
    try:
        if not os.path.exists(name):
            print(f"Đang tải {name}...")
            # Thêm Header để tránh bị GitHub chặn
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(name, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Xong: {name}")
        else:
            print(f"Đã có file: {name}")
    except Exception as e:
        print(f"Lỗi khi tải {name}: {e}")

print("--- Hoàn tất! Bạn có thể chạy website ngay bây giờ. ---")