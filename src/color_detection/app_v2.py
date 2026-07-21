import cv2
import os
import numpy as np
import math
import time
import logging
import socket
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

# --- 1. KHỞI TẠO HỆ THỐNG ---
folders = ['logs', 'runs', 'templates']
for f in folders:
    if not os.path.exists(f): os.makedirs(f)

log_filename = f"logs/robot_log_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', encoding='utf-8')

recent_logs = []


def log_event(msg):
    ts = datetime.now().strftime('%H:%M:%S');
    formatted = f"[{ts}] {msg}"
    logging.info(msg);
    recent_logs.insert(0, formatted)
    if len(recent_logs) > 25: recent_logs.pop()
    print(formatted)


# --- 2. CẤU HÌNH AI ---
app = Flask(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model_yolo = YOLO('color.pt')
detector_qr = cv2.wechat_qrcode_WeChatQRCode("detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")

CAMERA_INDEX = 0
SCAN_DURATION = 4
QR_MAP = {"hfuff": "Mã 1", "kztar": "Mã 2", "etply": "Mã 3"}
COLORS = ["red_box", "blue_box", "yellow_box"]

is_camera_on = False;
is_scanning = False;
is_confirmed = False
scan_start_time = 0;
countdown_val = 0;
current_fps = 0

temp_results = {};
task_queue = [];
qr_memory = {}
# scan_candidates lưu trữ: { 'red_box': [ {'qr': 'Mã 1', 'conf': 0.9, 'dist': 120}, ... ] }
scan_candidates = {cls: [] for cls in COLORS}
latest_frame = None


def normalize_qr_name(raw):
    name = str(raw).strip().lower()
    for k, v in QR_MAP.items():
        if k in name: return v
    return name


# --- 3. LOGIC XỬ LÝ FRAME ---
def process_frame(frame):
    global qr_memory, latest_frame, scan_candidates
    latest_frame = frame.copy()
    qr_res = detector_qr.detectAndDecode(frame)
    yolo_res = model_yolo.predict(frame, conf=0.6, verbose=False)  # Nâng conf lên 0.6 để chắc chắn

    if qr_res[0]:
        for i, name in enumerate(qr_res[0]):
            clean = normalize_qr_name(name)
            pts = qr_res[1][i].astype(int)
            center = (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))
            qr_memory[clean] = center  # Ghim vị trí vào bộ nhớ
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            cv2.putText(frame, clean, (pts[0][0], pts[0][1] - 15), 1, 2.0, (0, 255, 0), 3)

    for r in yolo_res:
        for box in r.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            cls = model_yolo.names[int(box.cls[0])]
            conf = float(box.conf[0])
            c_cx, c_cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2

            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (b[0], b[1] - 10), 1, 1.2, (255, 0, 0), 2)

            if is_scanning and cls in COLORS:
                for q_name, q_center in qr_memory.items():
                    dist = math.sqrt((c_cx - q_center[0]) ** 2 + (c_cy - q_center[1]) ** 2)
                    # Thu thập tất cả các cặp có khả năng trong bán kính 650px
                    if dist < 650:
                        scan_candidates[cls].append({"qr": q_name, "conf": conf, "dist": dist})
    return frame


def save_session(final_map, frame):
    session_id = datetime.now().strftime('session_%Y%m%d_%H%M%S')
    path = os.path.join('runs', session_id);
    os.makedirs(path)
    img_save = frame.copy()
    for i, (col, loc) in enumerate(final_map.items()):
        cv2.putText(img_save, f"{col}: {loc}", (50, 60 + i * 60), 1, 2.5, (0, 255, 255), 4)
    cv2.imwrite(os.path.join(path, f"{session_id}.jpg"), img_save)
    with open(os.path.join(path, 'results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"ID: {session_id}\nResults: {final_map}")
    log_event(f"📁 Lưu thành công tại: {path}")


# --- 4. STREAM & API ---
@app.route('/video_feed')
def video_feed():
    def gen():
        global is_scanning, scan_start_time, countdown_val, temp_results, scan_candidates, current_fps
        p_t = 0
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(3, 1280);
        cap.set(4, 720)
        while True:
            ret, frame = cap.read()
            if not ret: break
            t = time.time();
            current_fps = int(1 / (t - p_t)) if p_t != 0 else 0;
            p_t = t
            frame = process_frame(frame)
            if is_scanning:
                elap = time.time() - scan_start_time
                if elap < SCAN_DURATION:
                    countdown_val = round(SCAN_DURATION - elap, 1)
                else:
                    # --- THUẬT TOÁN TỐI ƯU TOÀN CỤC ---
                    all_matches = []
                    for c_name in COLORS:
                        for cand in scan_candidates[c_name]:
                            # Công thức điểm: Độ tin cậy cao / (Khoảng cách thấp + 1)
                            # Ưu tiên cực lớn cho khối có Conf cao nhất
                            score = (cand['conf'] ** 2 * 10000) / (cand['dist'] + 1)
                            all_matches.append({'color': c_name, 'qr': cand['qr'], 'score': score})

                    # Sắp xếp tất cả các cặp có thể theo điểm số từ cao xuống thấp
                    all_matches.sort(key=lambda x: x['score'], reverse=True)

                    voted = {};
                    used_qrs = set();
                    used_colors = set()
                    for m in all_matches:
                        if m['color'] not in used_colors and m['qr'] not in used_qrs:
                            voted[m['color']] = m['qr']
                            used_colors.add(m['color']);
                            used_qrs.add(m['qr'])

                    # Fallback: Đảm bảo đủ 3 vị trí nếu camera nhìn thiếu
                    rem_colors = [c for c in COLORS if c not in used_colors]
                    rem_qrs = [q for q in qr_memory.keys() if q not in used_qrs]
                    for i in range(min(len(rem_colors), len(rem_qrs))):
                        voted[rem_colors[i]] = rem_qrs[i]
                        log_event(f"Auto-Assign: {rem_colors[i]} -> {rem_qrs[i]}")

                    temp_results = voted;
                    is_scanning = False;
                    countdown_val = 0
                    log_event(f"Quét xong. Kết quả tối ưu: {temp_results}")
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index(): return render_template('index.html')


@app.route('/api/connect', methods=['POST'])
def connect():
    global is_camera_on;
    is_camera_on = True
    log_event("Camera Connected.");
    return jsonify({"status": "ok"})


@app.route('/api/start_scan', methods=['POST'])
def start_scan():
    global is_scanning, scan_start_time, scan_candidates, temp_results, is_confirmed
    if len(task_queue) > 0: return jsonify({"status": "error", "msg": "Hoàn thành phiên cũ!"})
    scan_candidates = {cls: [] for cls in COLORS}
    temp_results = {};
    is_confirmed = False;
    is_scanning = True;
    scan_start_time = time.time()
    log_event("Bắt đầu quét sa bàn...");
    return jsonify({"status": "ok"})


@app.route('/api/confirm', methods=['POST'])
def confirm():
    global task_queue, is_confirmed
    # Chốt thứ tự Đỏ -> Xanh -> Vàng
    task_queue = [{"color": c, "location": temp_results[c]} for c in COLORS if c in temp_results]
    is_confirmed = True;
    save_session(temp_results, latest_frame.copy())
    log_event(f"CONFIRMED: {task_queue}");
    return jsonify({"status": "ok"})


@app.route('/api/reset_session', methods=['POST'])
def reset_session():
    global task_queue, temp_results, is_confirmed, qr_memory, is_scanning
    task_queue = [];
    temp_results = {};
    is_confirmed = False;
    qr_memory = {};
    is_scanning = False
    log_event("--- RESET SYSTEM ---");
    return jsonify({"status": "ok"})


@app.route('/api/complete_task', methods=['POST'])
def complete():
    global task_queue
    if task_queue: log_event(f"Done: {task_queue.pop(0)['color']}"); return jsonify({"status": "ok"})


@app.route('/api/get_status', methods=['GET'])
def get_status():
    return jsonify(
        {"fps": current_fps, "countdown": countdown_val, "is_scanning": is_scanning, "temp_map": temp_results,
         "task_queue": task_queue, "is_confirmed": is_confirmed, "logs": recent_logs})


if __name__ == '__main__':
    ip = socket.gethostbyname(socket.gethostname())
    print(f"\n[SERVER] Dashboard: http://{ip}:5000\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)