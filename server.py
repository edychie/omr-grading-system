import os
import cv2
import numpy as np
import base64
import logging
import traceback
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# 設定日誌顯示詳細錯誤
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# ⭐ 設定 1：允許所有來源，並允許所有標頭 (最寬鬆模式)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# ⚙️ 參數設定
# ==========================================
INFO_X_START = 282
INFO_GAP = 128
INFO_Y_ADJ = 12
INFO_BOX_SIZE = 45

ANS_Y_ADJ = 22
ANS_GAP = 135
ANS_BOX_SIZE = 45

L_OFFSET = 282
M_OFFSET = 1018
R_OFFSET = 1774

PIXEL_THRESHOLD = 550

# ==========================================
# 🧠 核心邏輯
# ==========================================
def process_info_row(thresh_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    x_start = anchor[0] + offset
    y_start = anchor[1] + y_adj
    for i in range(10):
        x = x_start + (i * gap)
        if y_start < 0 or x < 0: continue
        roi = thresh_img[y_start:y_start+box_s, x:x+box_s]
        scores.append(cv2.countNonZero(roi))
    return scores.index(max(scores))

def process_answer_row(thresh_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    x_a = anchor[0]
    y_a = anchor[1] + y_adj
    for i in range(4):
        x = x_a + offset + (i * gap)
        if y_a < 0 or x < 0: 
            scores.append(0)
            continue
        roi = thresh_img[y_a:y_a+box_s, x:x+box_s]
        scores.append(cv2.countNonZero(roi))
        
    marked_indices = [idx for idx, s in enumerate(scores) if s > PIXEL_THRESHOLD]
    options = ['A', 'B', 'C', 'D']
    
    # ⭐⭐ 多選題修正區塊 ⭐⭐
    if len(marked_indices) == 0: 
        return "X"  # 沒畫記
    else: 
        # 將所有畫記的選項拼起來，例如畫了A和B就會回傳 "AB"
        return "".join([options[i] for i in marked_indices])

# 1. 修改 analyze_paper_simple 接收參數
def analyze_paper_simple(image, custom_params=None, debug_mode=False):
    try:
        # ... (前面的影像處理代碼) ...
        
        # 關鍵：檢查是否真的有找到定位點 (假設變數叫 anchors)
        if not anchors or len(anchors) == 0:
            return {"status": "error", "msg": "找不到答案卡的定位點，請確保圖片清晰且邊界完整。"}

        # ... (中間的辨識代碼) ...

        if debug_mode:
            # 產生預覽圖的邏輯
            return {"status": "success", "debug_image": base64_str}
            
        return {"status": "success", "answers": all_answers, "detected_seat": seat_num, ...}

    except Exception as e:
        # 如果發生任何未預期的錯誤，也回傳字典
        return {"status": "error", "msg": str(e)}
        
    def draw_roi(img, anchor, offset, gap, box_s, y_adj, count, color=(0, 255, 0)):
        for i in range(count):
            x = anchor[0] + offset + (i * gap)
            y = anchor[1] + y_adj
            cv2.rectangle(img, (x, y), (x + box_s, y + box_s), color, 3)

    if debug:
        # 畫出基本資料區
        for i in range(5):
            draw_roi(debug_img, anchors[i], p["INFO_X_START"], p["INFO_GAP"], p["INFO_BOX_SIZE"], p["INFO_Y_ADJ"], 10, (255, 0, 0))
        # 畫出答案區
        for i in range(5, 25):
            draw_roi(debug_img, anchors[i], p["L_OFFSET"], p["ANS_GAP"], p["ANS_BOX_SIZE"], p["ANS_Y_ADJ"], 4)
            draw_roi(debug_img, anchors[i], p["M_OFFSET"], p["ANS_GAP"], p["ANS_BOX_SIZE"], p["ANS_Y_ADJ"], 4)
            draw_roi(debug_img, anchors[i], p["R_OFFSET"], p["ANS_GAP"], p["ANS_BOX_SIZE"], p["ANS_Y_ADJ"], 4)

        _, buffer = cv2.imencode('.jpg', debug_img)
        debug_base64 = base64.b64encode(buffer).decode('utf-8')
        return {"debug_image": debug_base64}

    # ... 原本的分析邏輯回傳結果 ...
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
        img = decode_base64_image(image_data) # 假設你有這個輔助函式

        # 3. 呼叫辨識 (重點在這裡！)
        result = analyze_paper_simple(img, custom_params, debug_mode)

        # 預防措施：如果 analyze_paper_simple 因為意外回傳了 None
        if result is None:
            return jsonify({"status": "error", "msg": "後端處理程序回傳空值(None)"})

        # 4. 回傳結果
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "msg": f"伺服器內部錯誤: {str(e)}"})
