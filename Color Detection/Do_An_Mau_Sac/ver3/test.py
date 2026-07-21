# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
from picamera2 import Picamera2
import serial


# ================================
# GIAO TIẾP ARDUINO
# ================================
class ArduinoStepper:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            print(f"[INFO] Arduino connected on {port} at {baudrate} baud")
            self.ser.write(b"0,0\n")
            self.ser.flush()
        except Exception as e:
            self.ser = None
            print(f"[WARNING] Arduino connection failed: {e}")

    def send_frequency(self, freqL, freqR):
        if self.ser:
            freqL_safe = max(min(int(freqL), 5000), -5000)
            freqR_safe = max(min(int(freqR), 5000), -5000)
            cmd = f"{freqL_safe},{freqR_safe}\n"
            try:
                self.ser.write(cmd.encode())
                self.ser.flush()
            except Exception as e:
                print(f"[ERROR] Send failed: {e}")


# ================================
# PID CONTROLLER
# ================================
class PID:
    def __init__(self, kp, ki, kd, output_limits=(-500, 500)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_cte = 0.0
        self.integral = 0.0
        self.output_limits = output_limits

    def update(self, cte, dt=0.05):
        self.integral += cte * dt
        self.integral = max(min(self.integral, 1000), -1000)
        derivative = (cte - self.prev_cte) / (dt if dt > 1e-6 else 1e-6)
        out = self.kp * cte + self.ki * self.integral + self.kd * derivative
        out = max(min(out, self.output_limits[1]), self.output_limits[0])
        self.prev_cte = cte
        return out

    def reset(self):
        self.integral = 0.0
        self.prev_cte = 0.0


# ================================
# ROBUST LANE DETECTOR - KHÔNG BỊ NHẢY
# ================================
class RobustLaneDetector:
    def __init__(self, width=640, height=480, debug_draw=True):
        self.width = width
        self.height = height
        self.debug_draw = debug_draw

        # Sliding window parameters - TỐI ƯU
        self.nwindows = 9
        self.base_margin = 100
        self.margin_straight = 80
        self.margin_curve = 150
        self.minpix_strict = 40
        self.minpix_loose = 15

        # Lookahead distance
        self.lookahead_px = 50

        # QUAN TRỌNG: Lưu lịch sử polynomial fits
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.left_fit_history = []
        self.right_fit_history = []
        self.max_fit_history = 5

        # Lưu lịch sử lane centers
        self.prev_lane_centers = []
        self.prev_curvatures = []
        self.max_history = 8  # Giảm từ 10 → 8 để phản ứng nhanh hơn

        # Tracking lost frames
        self.left_lost_frames = 0
        self.right_lost_frames = 0
        self.max_lost_frames = 10

        # Lane width estimate (pixels)
        self.estimated_lane_width = self.width * 0.4
        self.lane_width_history = []

    def process_image(self, image):
        """Xử lý ảnh với CLAHE và Canny"""
        img = cv2.resize(image, (self.width, self.height))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge
        edges = cv2.Canny(blur, 50, 150)

        return img, edges

    def adaptive_margin(self, curvature):
        """Margin thay đổi theo độ cong"""
        if curvature < 500:
            return self.margin_curve
        elif curvature < 1000:
            return int(self.base_margin * 1.2)
        else:
            return self.margin_straight

    def adaptive_minpix(self, curvature):
        """Minpix thay đổi theo độ cong"""
        if curvature < 700:
            return self.minpix_loose
        else:
            return self.minpix_strict

    def smooth_histogram(self, histogram):
        """Làm mượt histogram trước khi tìm peak"""
        return cv2.GaussianBlur(histogram.reshape(1, -1), (21, 1), 0).flatten()

    def validate_fit(self, new_fit, prev_fit, max_change=0.001):
        """
        Kiểm tra fit mới có hợp lý không
        Nếu thay đổi quá lớn → blend với fit cũ
        """
        if prev_fit is None:
            return new_fit

        diff = np.abs(new_fit - prev_fit)

        # Nếu thay đổi quá lớn
        if np.max(diff) > max_change:
            # Blend: 70% mới + 30% cũ
            return 0.7 * new_fit + 0.3 * prev_fit

        return new_fit

    def predict_lane_from_opposite(self, fit, lane_width):
        """
        DỰ ĐOÁN vạch mất dựa vào vạch còn lại
        Ví dụ: Mất vạch trái, dùng vạch phải để dự đoán
        """
        if fit is None:
            return None

        # Tạo fit mới bằng cách shift theo lane_width
        predicted_fit = fit.copy()
        predicted_fit[2] = fit[2] - lane_width  # Shift c parameter

        return predicted_fit

    def fit_polynomial_robust(self, binary_img, prev_curvature=1000):
        """
        SLIDING WINDOW CẢI TIẾN - KHÔNG BỊ NHẢY
        """
        height = binary_img.shape[0]

        # Histogram với weighted sum (ưu tiên phần gần xe)
        weights = np.linspace(0.5, 2.0, height // 2)
        histogram = np.sum(binary_img[height // 2:, :] * weights[:, np.newaxis], axis=0)
        histogram = self.smooth_histogram(histogram)

        midpoint = len(histogram) // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Kiểm tra histogram peak có đủ mạnh không
        left_peak_strength = histogram[leftx_base]
        right_peak_strength = histogram[rightx_base]
        threshold = np.max(histogram) * 0.3

        left_detected = left_peak_strength > threshold
        right_detected = right_peak_strength > threshold

        # Adaptive parameters
        margin = self.adaptive_margin(prev_curvature)
        minpix = self.adaptive_minpix(prev_curvature)
        window_height = height // self.nwindows

        nonzero = binary_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # ============================================
        # SLIDING WINDOW CHO VẠCH TRÁI
        # ============================================
        leftx_current = leftx_base
        left_lane_inds = []
        left_centers = []

        for window in range(self.nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height

            # Adaptive margin: window cao hơn → margin lớn hơn
            window_margin = int(margin * (1 + window * 0.1))

            win_xleft_low = leftx_current - window_margin
            win_xleft_high = leftx_current + window_margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)

            # CẬP NHẬT TÂM WINDOW
            if len(good_left_inds) > minpix:
                left_x_points = nonzerox[good_left_inds]

                # Weighted mean: ưu tiên điểm gần tâm
                distances = np.abs(left_x_points - leftx_current)
                weights = np.exp(-distances / (window_margin / 2))
                leftx_new = int(np.average(left_x_points, weights=weights))

                # KIỂM TRA NHẢY: chỉ cập nhật nếu không nhảy quá xa
                jump = abs(leftx_new - leftx_current)
                if jump < 100:  # Giới hạn nhảy 100px
                    leftx_current = leftx_new
                # Nếu nhảy quá xa → giữ nguyên

                left_centers.append(leftx_current)
            else:
                # KHÔNG ĐỦ ĐIỂM: Dự đoán theo xu hướng
                if len(left_centers) >= 2:
                    trend = left_centers[-1] - left_centers[-2]
                    # Giới hạn trend
                    trend = max(min(trend, 15), -15)
                    leftx_current = left_centers[-1] + trend
                # Nếu không có lịch sử → giữ nguyên

                left_centers.append(leftx_current)

        # ============================================
        # SLIDING WINDOW CHO VẠCH PHẢI
        # ============================================
        rightx_current = rightx_base
        right_lane_inds = []
        right_centers = []

        for window in range(self.nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height

            window_margin = int(margin * (1 + window * 0.1))

            win_xright_low = rightx_current - window_margin
            win_xright_high = rightx_current + window_margin

            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            right_lane_inds.append(good_right_inds)

            if len(good_right_inds) > minpix:
                right_x_points = nonzerox[good_right_inds]
                distances = np.abs(right_x_points - rightx_current)
                weights = np.exp(-distances / (window_margin / 2))
                rightx_new = int(np.average(right_x_points, weights=weights))

                jump = abs(rightx_new - rightx_current)
                if jump < 100:
                    rightx_current = rightx_new

                right_centers.append(rightx_current)
            else:
                if len(right_centers) >= 2:
                    trend = right_centers[-1] - right_centers[-2]
                    trend = max(min(trend, 15), -15)
                    rightx_current = right_centers[-1] + trend

                right_centers.append(rightx_current)

        # ============================================
        # FIT POLYNOMIAL
        # ============================================
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = None
        right_fit = None

        # FIT VẠCH TRÁI
        if len(leftx) > 50 and left_detected:
            try:
                weights_left = np.linspace(1, 3, len(lefty))
                left_fit = np.polyfit(lefty, leftx, 2, w=weights_left)
                left_fit = self.validate_fit(left_fit, self.prev_left_fit)
                self.prev_left_fit = left_fit
                self.left_lost_frames = 0
            except:
                left_fit = self.prev_left_fit
        else:
            # MẤT VẠCH TRÁI
            self.left_lost_frames += 1

            # Nếu mới mất (< 5 frames) → dùng fit cũ
            if self.left_lost_frames < 5 and self.prev_left_fit is not None:
                left_fit = self.prev_left_fit
            # Nếu mất lâu nhưng có vạch phải → dự đoán từ vạch phải
            elif right_fit is not None:
                left_fit = self.predict_lane_from_opposite(right_fit, -self.estimated_lane_width)
            # Nếu không có gì → dùng fit cũ
            elif self.prev_left_fit is not None:
                left_fit = self.prev_left_fit

        # FIT VẠCH PHẢI
        if len(rightx) > 50 and right_detected:
            try:
                weights_right = np.linspace(1, 3, len(righty))
                right_fit = np.polyfit(righty, rightx, 2, w=weights_right)
                right_fit = self.validate_fit(right_fit, self.prev_right_fit)
                self.prev_right_fit = right_fit
                self.right_lost_frames = 0
            except:
                right_fit = self.prev_right_fit
        else:
            # MẤT VẠCH PHẢI
            self.right_lost_frames += 1

            # Nếu mới mất (< 5 frames) → dùng fit cũ
            if self.right_lost_frames < 5 and self.prev_right_fit is not None:
                right_fit = self.prev_right_fit
            # Nếu mất lâu nhưng có vạch trái → dự đoán từ vạch trái
            elif left_fit is not None:
                right_fit = self.predict_lane_from_opposite(left_fit, self.estimated_lane_width)
            # Nếu không có gì → dùng fit cũ
            elif self.prev_right_fit is not None:
                right_fit = self.prev_right_fit

        # CẬP NHẬT LANE WIDTH ESTIMATE
        if left_fit is not None and right_fit is not None:
            y_eval = height - self.lookahead_px
            left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
            right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
            current_width = abs(right_x - left_x)

            self.lane_width_history.append(current_width)
            if len(self.lane_width_history) > 10:
                self.lane_width_history.pop(0)
            self.estimated_lane_width = np.mean(self.lane_width_history)

        return left_fit, right_fit

    def calculate_lane_center(self, left_fit, right_fit):
        """Tính lane center với xử lý robust"""
        if left_fit is None and right_fit is None:
            return None, 0, "no_lane"

        y_eval = self.height - self.lookahead_px
        y_eval = max(0, min(y_eval, self.height - 1))

        if left_fit is not None:
            left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        else:
            left_x = None

        if right_fit is not None:
            right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        else:
            right_x = None

        # Xác định mode và tính lane center
        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2
            mode = "both_lanes"
        elif right_x is not None:
            lane_center = right_x - self.estimated_lane_width / 2
            mode = "right_only"
        elif left_x is not None:
            lane_center = left_x + self.estimated_lane_width / 2
            mode = "left_only"
        else:
            return None, 0, "no_lane"

        # Curvature
        curvature = 0
        if left_fit is not None:
            curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / abs(2 * left_fit[0] + 1e-6)
        elif right_fit is not None:
            curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / abs(2 * right_fit[0] + 1e-6)

        # Temporal filtering với trọng số giảm dần
        lane_center = self._apply_temporal_filter(lane_center)
        curvature = self._apply_curvature_filter(curvature)

        return lane_center, curvature, mode

    def _apply_temporal_filter(self, lane_center):
        """Temporal filter với trọng số exponential"""
        if lane_center is None:
            return None

        self.prev_lane_centers.append(lane_center)
        if len(self.prev_lane_centers) > self.max_history:
            self.prev_lane_centers.pop(0)

        # Exponential weighted average: frame gần hơn → trọng số cao hơn
        weights = np.exp(np.linspace(-1, 0, len(self.prev_lane_centers)))
        weights /= weights.sum()

        return np.average(self.prev_lane_centers, weights=weights)

    def _apply_curvature_filter(self, curvature):
        self.prev_curvatures.append(curvature)
        if len(self.prev_curvatures) > self.max_history:
            self.prev_curvatures.pop(0)
        return np.mean(self.prev_curvatures)

    def detect_lane(self, frame):
        """Main detection"""
        img, edges = self.process_image(frame)

        # Estimate curvature từ frame trước
        prev_curvature = self.prev_curvatures[-1] if self.prev_curvatures else 1000

        # Fit polynomial
        left_fit, right_fit = self.fit_polynomial_robust(edges, prev_curvature)

        # Calculate lane center
        lane_center, curvature, mode = self.calculate_lane_center(left_fit, right_fit)

        # Visualization
        if self.debug_draw:
            result = img.copy()

            # Draw fitted curves
            if left_fit is not None or right_fit is not None:
                ploty = np.linspace(0, self.height - 1, self.height)

                if left_fit is not None:
                    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
                    left_fitx = np.clip(left_fitx, 0, self.width - 1)
                    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)

                    # Màu khác nhau nếu đang dự đoán
                    color = (255, 0, 0) if self.left_lost_frames == 0 else (255, 0, 255)
                    cv2.polylines(result, pts_left, False, color, 3)

                if right_fit is not None:
                    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
                    right_fitx = np.clip(right_fitx, 0, self.width - 1)
                    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)

                    color = (0, 0, 255) if self.right_lost_frames == 0 else (255, 0, 255)
                    cv2.polylines(result, pts_right, False, color, 3)

            # Draw lane center và lookahead
            if lane_center is not None:
                y_look = self.height - self.lookahead_px
                cv2.circle(result, (int(lane_center), y_look), 10, (0, 255, 0), -1)
                cv2.line(result, (self.width // 2, y_look), (int(lane_center), y_look), (0, 255, 255), 3)
                cv2.line(result, (0, y_look), (self.width, y_look), (255, 255, 0), 1)

            # Draw center line
            cv2.line(result, (self.width // 2, 0), (self.width // 2, self.height), (255, 255, 0), 1)

            # Status text
            if self.left_lost_frames > 0:
                cv2.putText(result, f"L-PREDICT ({self.left_lost_frames})",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            if self.right_lost_frames > 0:
                cv2.putText(result, f"R-PREDICT ({self.right_lost_frames})",
                            (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        else:
            result = img

        return result, edges, lane_center, curvature, mode


# ================================
# ADAPTIVE SPEED CONTROLLER
# ================================
class AdaptiveSpeedController:
    def __init__(self, base_freq=1500, min_freq=800):
        self.base_freq = base_freq
        self.min_freq = min_freq

    def get_speed(self, curvature, error):
        if curvature < 500:
            speed_factor = 0.5
        elif curvature < 1000:
            speed_factor = 0.7
        else:
            speed_factor = 1.0

        if abs(error) > 100:
            speed_factor *= 0.8
        elif abs(error) > 150:
            speed_factor *= 0.6

        target_freq = int(self.base_freq * speed_factor)
        return max(target_freq, self.min_freq)


# ================================
# LANE FOLLOWER
# ================================
class LaneFollower:
    def __init__(self, pid, base_freq=1500):
        self.pid = pid
        self.base_freq = base_freq
        self.speed_controller = AdaptiveSpeedController(base_freq)
        self.last_valid_center = None
        self.frames_lost = 0
        self.max_lost_frames = 15

    def compute_speeds(self, lane_center, curvature, mode, image_center):
        if mode in ("both_lanes", "left_only", "right_only"):
            self.frames_lost = 0
            self.last_valid_center = lane_center

            error = (lane_center - image_center) if lane_center is not None else 0
            current_base = self.speed_controller.get_speed(curvature, error)
            omega = self.pid.update(error)

            freqL = current_base - omega
            freqR = current_base + omega

            status = f"{mode} | Curve:{int(curvature)} | Speed:{int(current_base)}"

        elif mode == "no_lane":
            self.frames_lost += 1
            if self.frames_lost < self.max_lost_frames and self.last_valid_center:
                error = self.last_valid_center - image_center
                omega = self.pid.update(error) * 0.5
                freqL = int(self.base_freq * 0.5 - omega)
                freqR = int(self.base_freq * 0.5 + omega)
                status = f"Lost ({self.frames_lost}/{self.max_lost_frames})"
            else:
                self.pid.reset()
                freqL = -int(self.base_freq * 0.3)
                freqR = int(self.base_freq * 0.3)
                status = "Searching..."
        else:
            freqL = 0
            freqR = 0
            status = f"Unknown: {mode}"

        return freqL, freqR, status


# ================================
# MAIN
# ================================
def main():
    # Parameters
    pid = PID(kp=0.8, ki=0.003, kd=0.05, output_limits=(-1000, 1000))
    BASE_FREQ = 1500
    MAX_FREQ = 3000
    MIN_FREQ = -3000

    # Hardware
    arduino = ArduinoStepper(port='/dev/ttyUSB0', baudrate=115200)
    lane_detector = RobustLaneDetector(width=640, height=480, debug_draw=True)
    lane_follower = LaneFollower(pid, base_freq=BASE_FREQ)

    # Camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    last_send_time = 0.0
    send_period = 0.1

    print("[INFO] ========================================")
    print("[INFO] ROBUST Lane Following - NO JUMP Version")
    print("[INFO] ========================================")
    print("[INFO] Features:")
    print("[INFO] - 9 sliding windows (optimal)")
    print("[INFO] - Adaptive margin & minpix")
    print("[INFO] - Jump detection (max 100px/window)")
    print("[INFO] - Trend prediction when lost")
    print("[INFO] - Lane prediction from opposite side")
    print("[INFO] - Exponential weighted temporal filter")
    print("[INFO] Press 'q' to quit")
    print("[INFO] ========================================")

    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                continue

            result, edges, lane_center, curvature, mode = lane_detector.detect_lane(frame)
            image_center = result.shape[1] // 2

            # Compute speeds
            freqL, freqR, status = lane_follower.compute_speeds(lane_center, curvature, mode, image_center)

            # Clip
            freqL = int(np.clip(freqL, MIN_FREQ, MAX_FREQ))
            freqR = int(np.clip(freqR, MIN_FREQ, MAX_FREQ))

            # Send
            now = time.time()
            if now - last_send_time >= send_period:
                arduino.send_frequency(freqL, freqR)
                last_send_time = now

            # Display
            error = int(lane_center - image_center) if lane_center else 0
            cv2.putText(result, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f"Err:{error} L:{freqL}Hz R:{freqR}Hz",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, "ROBUST MODE - NO JUMP", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            cv2.imshow("Robust Lane Detection", result)
            cv2.imshow("Edges", edges)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        print("[INFO] Stopping...")
        for _ in range(3):
            arduino.send_frequency(0, 0)
            time.sleep(0.1)
            
if __name__ == "__main__":
    main()
