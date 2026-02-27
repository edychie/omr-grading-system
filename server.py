import os
import cv2
import numpy as np
import base64
import logging
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# ⚙️ 參數設定 (維持不變)
# ==========================================
PIXEL_THRESHOLD = 330
# ... (其餘參數維持不變) ...

# ==========================================
# 🧠 核心邏輯 (加入畫圖功能)
# ==========================================

def process_info_row(thresh_img, debug_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    x_start = anchor[0] + offset
    y_start = anchor[1] + y_adj
    best_idx = -1
    max_score = -1
    
    for i in range(10):
        x = x_start + (i * gap)
        roi = thresh_img[y_start:y_start+box_s, x:x+box_s]
        score = cv2.countNonZero(roi)
        scores.append(score)
        # 畫出掃描範圍 (藍色細框)
        cv2.rectangle(debug_img, (x, y_start), (x+box_s, y_start+box_s), (255, 0, 0), 2)
        if score > max_score:
            max_score = score
            best_idx = i
            
    # 將判定選擇的項目畫上綠色粗框
    target_x = x_start + (best_idx * gap)
    cv2.rectangle(debug_img, (target_x, y_start), (target_x+box_s, y_start+box_s), (0, 255, 0), 5)
    return best_idx

def process_answer_row(thresh_img, debug_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    x_a = anchor[0]
    y_a = anchor[1] + y_adj
    options = ['A', 'B', 'C', 'D']
    marked_indices = []
    
    for i in range(4):
        x = x_a + offset + (i * gap)
        roi = thresh_img[y_a:y_a+box_s, x:x+box_s]
        score = cv2.countNonZero(roi)
        # 畫出選項掃描區 (藍色細框)
        cv2.rectangle(debug_img, (x, y_a), (x+box_s, y_a+box_s), (255, 0, 0), 2)
        
        if score > PIXEL_THRESHOLD:
            marked_indices.append(i)
            # 判定有塗黑的畫上綠色粗框
            cv2.rectangle(debug_img, (x, y_a), (x+box_s, y_a+box_s), (0, 255, 0), 5)
            
    return "".join([options[idx] for idx in marked_indices])

def analyze_paper_simple(image):
    target_size = (2480, 3508)
    if image.shape[:2] != (target_size[1], target_size[0]):
        image = cv2.resize(image, target_size)
    
    # 建立一張用於畫框的彩色副本
    debug_img = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 11)
    
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anchors = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < 150 and 20 < w < 80 and 0.8 < (w/h) < 1.2:
            anchors.append((x, y, w, h))
            # 畫出定位點 (黃色)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 255), 3)
            
    anchors = sorted(anchors, key=lambda b: b[1])
    if len(anchors) < 25: raise Exception("定位點不足")

    # 處理個人資訊 (傳入 debug_img)
    grade = process_info_row(thresh_inv, debug_img, anchors[0], 282, 128, 45, 12)
    c1 = process_info_row(thresh_inv, debug_img, anchors[1], 282, 128, 45, 12)
    c2 = process_info_row(thresh_inv, debug_img, anchors[2], 282, 128, 45, 12)
    s1 = process_info_row(thresh_inv, debug_img, anchors[3], 282, 128, 45, 12)
    s2 = process_info_row(thresh_inv, debug_img, anchors[4], 282, 128, 45, 12)
    
    # 處理答案
    ans_list = [""] * 60
    offsets = [282, 1018, 1774]
    for i in range(5, 25):
        for col in range(3):
            idx = (i-5) + (col * 20)
            ans_list[idx] = process_answer_row(thresh_inv, debug_img, anchors[i], offsets[col], 135, 45, 22)

    # 將畫好框的 debug_img 縮小 (避免 Base64 太大) 並編碼
    small_debug = cv2.resize(debug_img, (800, 1131)) # 縮小成較好顯示的大小
    _, buffer = cv2.imencode('.jpg', small_debug, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    debug_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "grade": str(grade), "class_name": f"{c1}{c2}", "seat": f"{s1}{s2}",
        "answers": ans_list,
        "debug_image": debug_base64
    }

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        image_base64 = data.get('image').split(",")[1] if "," in data.get('image') else data.get('image')
        img_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        result = analyze_paper_simple(image)
        
        return jsonify({
            "status": "success",
            "answers": result["answers"],
            "detected_grade": result["grade"],
            "detected_class": result["class_name"],
            "detected_seat": result["seat"],
            "debug_image": "data:image/jpeg;base64," + result["debug_image"] # 帶上前綴
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
