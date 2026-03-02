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
def analyze_paper_simple(image, params=None, debug=False):
    # 如果沒傳參數，使用預設值
    p = {
        "INFO_X_START": 282, "INFO_GAP": 128, "INFO_Y_ADJ": 12, "INFO_BOX_SIZE": 45,
        "ANS_Y_ADJ": 22, "ANS_GAP": 135, "ANS_BOX_SIZE": 45,
        "L_OFFSET": 282, "M_OFFSET": 1018, "R_OFFSET": 1774
    }
    if params: p.update(params)

    target_size = (2480, 3508)
    if image.shape[:2] != (target_size[1], target_size[0]):
        image = cv2.resize(image, target_size)
    
    # 建立一個畫框框用的備份圖
    debug_img = image.copy() if debug else None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 1)
    
    # ... (中間找 anchors 的程式碼不變) ...

    # 在畫框框的邏輯中 (範例：答案區)
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
        if not data:
            return jsonify({"status": "error", "msg": "沒有收到 JSON 資料"}), 400
            
        image_base64 = data.get('image')
        if not image_base64: 
            return jsonify({"status": "error", "msg": "沒有收到 image 欄位"}), 400
        
        # 處理 Base64 前綴 (如果有)
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        try:
            img_data = base64.b64decode(image_base64)
            np_arr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                 return jsonify({"status": "error", "msg": "圖片解碼失敗，可能是格式錯誤"}), 400
        except Exception as e:
            return jsonify({"status": "error", "msg": f"Base64 轉換錯誤: {str(e)}"}), 400
        
        result = analyze_paper_simple(image)
        
        response = jsonify({
            "status": "success",
            "answers": result["answers"],
            "detected_grade": result["grade"],
            "detected_class": result["class_name"],
            "detected_seat": result["seat"]
        })
        # ⭐ 設定 3：在回應中再次強制加入 Header
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        # 把詳細錯誤印在 Render Logs 裡
        print(f"🔥 嚴重錯誤: {e}") 
        traceback.print_exc()
        
        response = jsonify({"status": "error", "msg": f"伺服器內部錯誤: {str(e)}"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

