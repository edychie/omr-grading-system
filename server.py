@@ -2,33 +2,34 @@
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS  # ⭐ 新增這一行：引入 CORS 套件
import logging
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# 設定日誌顯示詳細錯誤
logging.basicConfig(level=logging.DEBUG)

# 初始化 Flask
app = Flask(__name__)
CORS(app)  # ⭐ 新增這一行：允許跨網域存取 (解決 Failed to fetch)

# ⭐ 設定 1：允許所有來源，並允許所有標頭 (最寬鬆模式)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# ⚙️ 參數設定
# ==========================================
# 1. 學生資訊區 (藍色)
INFO_X_START = 282
INFO_GAP = 128
INFO_Y_ADJ = 12
INFO_BOX_SIZE = 45

# 2. 作答區 (綠色)
ANS_Y_ADJ = 22
ANS_GAP = 135
ANS_BOX_SIZE = 45

# 三欄位置
L_OFFSET = 282
M_OFFSET = 1018
R_OFFSET = 1774

# 判定黑度的門檻
PIXEL_THRESHOLD = 550

# ==========================================
@@ -38,132 +39,124 @@ def process_info_row(thresh_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    x_start = anchor[0] + offset
    y_start = anchor[1] + y_adj
    
    for i in range(10):
        x = x_start + (i * gap)
        if y_start < 0 or x < 0: continue
        roi = thresh_img[y_start:y_start+box_s, x:x+box_s]
        score = cv2.countNonZero(roi)
        scores.append(score)
        
        scores.append(cv2.countNonZero(roi))
    return scores.index(max(scores))

def process_answer_row(thresh_img, anchor, offset, gap, box_s, y_adj):
    scores = []
    x_a = anchor[0]
    y_a = anchor[1] + y_adj
    
    for i in range(4): # ABCD
    for i in range(4):
        x = x_a + offset + (i * gap)
        if y_a < 0 or x < 0: 
            scores.append(0)
            continue
        roi = thresh_img[y_a:y_a+box_s, x:x+box_s]
        scores.append(cv2.countNonZero(roi))

    marked_indices = [idx for idx, s in enumerate(scores) if s > PIXEL_THRESHOLD]
    options = ['A', 'B', 'C', 'D']
    
    if len(marked_indices) == 0: return "X"
    elif len(marked_indices) > 1: return "M"
    else: return options[marked_indices[0]]

def analyze_paper_simple(image):
    # 強制鎖定尺寸
    target_size = (2480, 3508)
    if image.shape[:2] != (target_size[1], target_size[0]):
        image = cv2.resize(image, target_size)

    # 轉灰階 & 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 35, 1
    )
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 1)

    # 找定位點
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anchors = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < 150 and 20 < w < 80 and 0.8 < (w/h) < 1.2:
            anchors.append((x, y, w, h))
    
    anchors = sorted(anchors, key=lambda b: b[1])

    if len(anchors) < 25:
        # 如果定位點不夠，拋出錯誤
    if len(anchors) < 25: 
        raise Exception(f"定位點不足 (只找到 {len(anchors)} 個，需要 25 個)")

    # 解析內容
    try:
        grade = process_info_row(thresh_inv, anchors[0], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        c1 = process_info_row(thresh_inv, anchors[1], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        c2 = process_info_row(thresh_inv, anchors[2], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        s1 = process_info_row(thresh_inv, anchors[3], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
        s2 = process_info_row(thresh_inv, anchors[4], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)

        # 組合 60 題答案
        ans_list = [""] * 60
        for i in range(5, 25):
            # 左欄 (1-20)
            ans_list[i-5] = process_answer_row(thresh_inv, anchors[i], L_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
            # 中欄 (21-40)
            ans_list[i-5+20] = process_answer_row(thresh_inv, anchors[i], M_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
            # 右欄 (41-60)
            ans_list[i-5+40] = process_answer_row(thresh_inv, anchors[i], R_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
            
        full_answers = "".join(ans_list)

        return {
            "grade": str(grade),
            "class_name": f"{c1}{c2}",
            "seat": f"{s1}{s2}",
            "answers": full_answers
        }
        
    except Exception as e:
        raise Exception(f"解析過程錯誤: {str(e)}")
    grade = process_info_row(thresh_inv, anchors[0], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
    c1 = process_info_row(thresh_inv, anchors[1], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
    c2 = process_info_row(thresh_inv, anchors[2], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
    s1 = process_info_row(thresh_inv, anchors[3], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)
    s2 = process_info_row(thresh_inv, anchors[4], INFO_X_START, INFO_GAP, INFO_BOX_SIZE, INFO_Y_ADJ)

    ans_list = [""] * 60
    for i in range(5, 25):
        ans_list[i-5] = process_answer_row(thresh_inv, anchors[i], L_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
        ans_list[i-5+20] = process_answer_row(thresh_inv, anchors[i], M_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
        ans_list[i-5+40] = process_answer_row(thresh_inv, anchors[i], R_OFFSET, ANS_GAP, ANS_BOX_SIZE, ANS_Y_ADJ)
    
    return {
        "grade": str(grade),
        "class_name": f"{c1}{c2}",
        "seat": f"{s1}{s2}",
        "answers": "".join(ans_list)
    }

# ⭐ 設定 2：手動處理 OPTIONS 請求 (確保萬無一失)
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

# ==========================================
# 🚀 接收圖片的 API 入口 (Flask)
# ==========================================
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "msg": "沒有收到 JSON 資料"}), 400
            
        image_base64 = data.get('image')
        if not image_base64: 
            return jsonify({"status": "error", "msg": "沒有收到 image 欄位"}), 400

        if not image_base64:
            return jsonify({"status": "error", "msg": "沒有收到圖片"}), 400

        # 1. 解碼圖片 (Base64 -> OpenCV)
        img_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
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

        # 2. 執行您的辨識邏輯
        result = analyze_paper_simple(image)

        # 3. 回傳完整 JSON 給 GAS
        return jsonify({
        response = jsonify({
            "status": "success",
            "answers": result["answers"],      # 60題答案
            "detected_grade": result["grade"], # 辨識到的年級
            "detected_class": result["class_name"], # 辨識到的班級
            "detected_seat": result["seat"]    # 辨識到的座號
            "answers": result["answers"],
            "detected_grade": result["grade"],
            "detected_class": result["class_name"],
            "detected_seat": result["seat"]
        })
        
        # ⭐ 設定 3：在回應中再次強制加入 Header
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        print(f"錯誤: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500
        # 把詳細錯誤印在 Render Logs 裡
        print(f"🔥 嚴重錯誤: {e}") 
        import traceback
        traceback.print_exc()
        
        response = jsonify({"status": "error", "msg": f"伺服器內部錯誤: {str(e)}"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500

# ==========================================
# 🌟 Render 啟動點
# ==========================================
if __name__ == '__main__':
    # 雲端 Render 會使用 gunicorn 啟動，不會執行這裡
    # 這裡的代碼僅供您在本地電腦測試使用
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

    app.run(host='0.0.0.0', port=port)
