import os
import cv2
import numpy as np
import base64
import logging
import traceback
import math
from collections import Counter
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# 設定日誌顯示詳細錯誤
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# ⭐ 設定 1：允許所有來源，並允許所有標頭 (最寬鬆模式)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# ⚙️ 參數設定 (整合絕對像素數值與校正參數預設值)
# ==========================================
DEFAULT_PARAMS = {
    # 影像處理靈敏度
    "PIXEL_THRESHOLD": 330,   # 判斷是否塗黑的像素門檻
    "BINARY_C": 11,           # 局部二值化的常數
    "ANCHOR_THRESH": 200,     # 尋找定位點的二值化門檻
    "TARGET_WIDTH": 2000,     # 校正後的統一寬度 (確保絕對像素對得齊)

    # 基本資料區 (絕對像素)
    "INFO_X_START": 282, 
    "INFO_GAP": 128, 
    "INFO_Y_ADJ": 12, 
    "INFO_BOX_SIZE": 45,

    # 作答區 (絕對像素)
    "L_OFFSET": 282,
    "M_OFFSET": 1018,
    "R_OFFSET": 1774,
    "ANS_GAP": 135,
    "ANS_BOX_SIZE": 45,
    "ANS_Y_ADJ": 22
}

# ==========================================
# 🧠 輔助與核心邏輯
# ==========================================

def decode_base64_image(base64_str):
    """將前端傳來的 Base64 字串轉回 OpenCV 格式的圖片"""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.error(f"解碼圖片失敗: {str(e)}")
        return None

def get_true_anchor_column(contours, max_x):
    """鎖定真正的定位點垂直列，過濾所有雜訊"""
    temp_anchors = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 條件：在左半部、寬度足夠、長寬比接近正方形
        if x < max_x and w > 12 and 0.6 < (w/h) < 1.4:
            if cv2.contourArea(cnt) / (w*h) > 0.4:
                temp_anchors.append((x, y, w, h))

    if not temp_anchors:
        return []

    # 找出最常見的 X 座標 (將 X 座標四捨五入到十位數來分群)
    x_bins = [round(a[0], -1) for a in temp_anchors]
    most_common_x = Counter(x_bins).most_common(1)[0][0]
    true_anchors = [a for a in temp_anchors if abs(a[0] - most_common_x) < 30]
    
    # 依照 Y 座標由上到下排序
    return sorted(true_anchors, key=lambda b: b[1])

def auto_align_and_crop(image, p):
    """自動旋轉校正、邊界偵測與裁切縮放"""
    h_raw, w_raw = image.shape[:2]

    # --- 步驟 1: 旋轉校正 ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, p["ANCHOR_THRESH"], 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    true_anchors = get_true_anchor_column(contours, w_raw // 2)

    if len(true_anchors) >= 5:
        points = np.array([[a[0] + a[2]//2, a[1] + a[3]//2] for a in true_anchors])
        [vx, vy, fit_x, fit_y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

        angle = math.degrees(math.atan2(float(vy[0]), float(vx[0])))
        correction_angle = 90 - angle
        if correction_angle > 45: correction_angle -= 180
        if correction_angle < -45: correction_angle += 180

        if 0.1 < abs(correction_angle) < 15:
            center = (w_raw // 2, h_raw // 2)
            M = cv2.getRotationMatrix2D(center, -correction_angle, 1.0)
            cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
            new_w = int((h_raw * sin) + (w_raw * cos))
            new_h = int((h_raw * cos) + (w_raw * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            image = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # --- 步驟 2: 邊界尋找與強制縮放至 TARGET_WIDTH ---
    gray_r = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_r = cv2.threshold(gray_r, p["ANCHOR_THRESH"], 255, cv2.THRESH_BINARY_INV)
    contours_r, _ = cv2.findContours(thresh_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_anchors = get_true_anchor_column(contours_r, image.shape[1] // 2)

    if len(final_anchors) >= 5:
        column_x = int(np.median([a[0] for a in final_anchors]))
        first_anchor_y = final_anchors[0][1]

        start_x = max(0, column_x - 50)
        start_y = max(0, first_anchor_y - 50)

        # 鎖定裁切寬度 (固定比例，讓絕對偏移量可以準確對應)
        FIXED_CROP_WIDTH = 1850 
        end_x = min(image.shape[1], start_x + FIXED_CROP_WIDTH)
        end_y = image.shape[0]

        cropped_img = image[start_y:end_y, start_x:end_x]
        
        # 強制縮放寬度至目標設定值 (例如 2000)，高度等比
        h_crop, w_crop = cropped_img.shape[:2]
        if h_crop > 0 and w_crop > 0:
            scale = p["TARGET_WIDTH"] / float(w_crop)
            final_h = int(h_crop * scale)
            image = cv2.resize(cropped_img, (p["TARGET_WIDTH"], final_h), interpolation=cv2.INTER_LANCZOS4)

    return image

def process_info_row(thresh_img, debug_img, anchor, offset, gap, box_s, y_adj, p_thresh, debug_mode):
    """處理基本資料列 (年級、班級、座號)"""
    scores = []
    x_start = anchor[0] + offset
    y_start = anchor[1] + y_adj

    for i in range(10):
        x = x_start + (i * gap)
        if y_start < 0 or x < 0: 
            scores.append(0)
            continue
            
        roi = thresh_img[y_start:y_start+box_s, x:x+box_s]
        score = cv2.countNonZero(roi)
        scores.append(score)

        if debug_mode:
            color = (0, 255, 0) if score > p_thresh else (0, 0, 255)
            cv2.rectangle(debug_img, (x, y_start), (x+box_s, y_start+box_s), color, 2)

    return str(scores.index(max(scores))) if max(scores) > p_thresh else "?"

def process_answer_row(thresh_img, debug_img, anchor, offset, gap, box_s, y_adj, p_thresh, debug_mode):
    """處理作答區列 (多選題修正邏輯)"""
    scores = []
    x_a = anchor[0]
    y_a = anchor[1] + y_adj

    for i in range(4):
        x = x_a + offset + (i * gap)
        if y_a < 0 or x < 0: 
            scores.append(0)
            continue
            
        roi = thresh_img[y_a:y_a+box_s, x:x+box_s]
        score = cv2.countNonZero(roi)
        scores.append(score)

        if debug_mode:
            color = (0, 255, 0) if score > p_thresh else (0, 0, 255)
            cv2.rectangle(debug_img, (x, y_a), (x+box_s, y_a+box_s), color, 2)
        
    marked_indices = [idx for idx, s in enumerate(scores) if s > p_thresh]
    options = ['A', 'B', 'C', 'D']
    
    # ⭐⭐ 多選題修正區塊 ⭐⭐
    if len(marked_indices) == 0: 
        ans = "X"  # 沒畫記
    else: 
        # 將所有畫記的選項拼起來，例如畫了A和B就會回傳 "AB"
        ans = "".join([options[i] for i in marked_indices])
        
    if debug_mode:
        cv2.putText(debug_img, ans, (x_a + offset - 40, y_a + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
    return ans

# ==========================================
# 🚀 核心辨識主程式
# ==========================================

def analyze_paper_simple(image, custom_params=None, debug_mode=False):
    try:
        # 1. 參數設定：整合預設值與前端傳來的值
        p = DEFAULT_PARAMS.copy()
        if custom_params and isinstance(custom_params, dict):
            p.update(custom_params)

        # 2. 自動校正與裁切
        aligned_image = auto_align_and_crop(image, p)
        debug_img = aligned_image.copy() if debug_mode else None

        # 3. 灰階與自適應二值化 (過濾陰影)
        gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        thresh_inv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 75, p["BINARY_C"]
        )

        # 4. 尋找並過濾定位點
        contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_anchors = get_true_anchor_column(contours, aligned_image.shape[1] // 3)

        # 確保定位點不重複 (⭐ Y軸距離改為大於 30 才算新的定位點，避免高解析度誤判)
        anchors = []
        if raw_anchors:
            anchors.append(raw_anchors[0])
            for i in range(1, len(raw_anchors)):
                if raw_anchors[i][1] - anchors[-1][1] > 30:
                    anchors.append(raw_anchors[i])

        if len(anchors) < 25:
            return {"status": "error", "msg": f"定位點不足 (找到 {len(anchors)} 個，需要至少 25 個)"}

        # 5. 辨識學生資料 (前5個定位點)
        grade = process_info_row(thresh_inv, debug_img, anchors[0], p["INFO_X_START"], p["INFO_GAP"], p["INFO_BOX_SIZE"], p["INFO_Y_ADJ"], p["PIXEL_THRESHOLD"], debug_mode)
        c1 = process_info_row(thresh_inv, debug_img, anchors[1], p["INFO_X_START"], p["INFO_GAP"], p["INFO_BOX_SIZE"], p["INFO_Y_ADJ"], p["PIXEL_THRESHOLD"], debug_mode)
        c2 = process_info_row(thresh_inv, debug_img, anchors[2], p["INFO_X_START"], p["INFO_GAP"], p["INFO_BOX_SIZE"], p["INFO_Y_ADJ"], p["PIXEL_THRESHOLD"], debug_mode)
        s1 = process_info_row(thresh_inv, debug_img, anchors[3], p["INFO_X_START"], p["INFO_GAP"], p["INFO_BOX_SIZE"], p["INFO_Y_ADJ"], p["PIXEL_THRESHOLD"], debug_mode)
        s2 = process_info_row(thresh_inv, debug_img, anchors[4], p["INFO_X_START"], p["INFO_GAP"], p["INFO_BOX_SIZE"], p["INFO_Y_ADJ"], p["PIXEL_THRESHOLD"], debug_mode)

        # 6. 辨識答案區 (第6~25個定位點)
        ans_list = [""] * 60
        current_idx = 5
        
        for i in range(20):
            if current_idx >= len(anchors): break
            anchor = anchors[current_idx]
            current_idx += 1

            # 左邊 (題號 1~20)
            ans_list[i] = process_answer_row(thresh_inv, debug_img, anchor, p["L_OFFSET"], p["ANS_GAP"], p["ANS_BOX_SIZE"], p["ANS_Y_ADJ"], p["PIXEL_THRESHOLD"], debug_mode)
            # 中間 (題號 21~40)
            ans_list[i + 20] = process_answer_row(thresh_inv, debug_img, anchor, p["M_OFFSET"], p["ANS_GAP"], p["ANS_BOX_SIZE"], p["ANS_Y_ADJ"], p["PIXEL_THRESHOLD"], debug_mode)
            # 右邊 (題號 41~60)
            ans_list[i + 40] = process_answer_row(thresh_inv, debug_img, anchor, p["R_OFFSET"], p["ANS_GAP"], p["ANS_BOX_SIZE"], p["ANS_Y_ADJ"], p["PIXEL_THRESHOLD"], debug_mode)

        # 7. 組織回傳結果
        response_data = {
            "status": "success", 
            "answers": "".join(ans_list), 
            "detected_grade": grade, 
            "detected_class": f"{c1}{c2}", 
            "detected_seat": f"{s1}{s2}"
        }

        # 8. 如果是「偵測校正模式」，附上 Base64 預覽圖
        if debug_mode and debug_img is not None:
            # 畫出定位點紅框示意
            for a in anchors:
                cv2.rectangle(debug_img, (a[0], a[1]), (a[0]+a[2], a[1]+a[3]), (0, 0, 255), 2)
                
            _, buffer = cv2.imencode('.jpg', debug_img)
            debug_b64 = base64.b64encode(buffer).decode('utf-8')
            response_data["debug_image"] = debug_b64

        return response_data

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "msg": f"辨識過程出錯: {str(e)}"}

# ==========================================
# 🌐 API 路由
# ==========================================

# ⭐ 設定 2：手動處理 OPTIONS 請求 (確保萬無一失)
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"status": "error", "msg": "沒有收到圖片資料"})

        # 1. 取得圖片與參數
        image_data = data.get('image')
        custom_params = data.get('params')
        debug_mode = data.get('debug_mode', False)

        # 2. 解碼圖片
        img = decode_base64_image(image_data)
        if img is None:
            return jsonify({"status": "error", "msg": "圖片解碼失敗，請確認 Base64 格式"})

        # 3. 呼叫辨識
        result = analyze_paper_simple(img, custom_params, debug_mode)

        if result is None:
            return jsonify({"status": "error", "msg": "後端處理程序回傳空值(None)"})

        # 4. 回傳結果
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "msg": f"伺服器內部錯誤: {str(e)}"})

if __name__ == '__main__':
    # 適合直接本地端或部署環境測試執行
    app.run(host='0.0.0.0', port=5000)
    # 適合直接本地端或部署環境測試執行
    app.run(host='0.0.0.0', port=5000)




