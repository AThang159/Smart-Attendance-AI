import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model', 'efficientNetB0.keras')
model = load_model(MODEL_PATH)

# Nhãn lớp
class_labels = [
    '21060451_NguyenHungAnh',
    '21090261_DuongNgocAnh',
    '21094341_ChauTieuLong',
    '21096911_NguyenNhatTung',
    '21105351_TongThanhLoc',
    '21119631_NguyenMinhLong'
]

# Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Hàm xử lý ảnh đầu vào
def extract_face(image_cv2, input_size=(224, 224)):
    gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, "Không tìm thấy khuôn mặt nào."

    # Chọn khuôn mặt đầu tiên
    (x, y, w, h) = faces[0]
    face_img = image_cv2[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Resize + chuẩn hóa
    img = Image.fromarray(face_rgb).resize(input_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array, None

@app.route('/face-detect', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({"message": "Không có ảnh trong yêu cầu!"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"message": "Chưa chọn ảnh!"}), 400

    try:
        # Chuyển ảnh sang OpenCV format
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image_cv2 = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Trích xuất khuôn mặt
        img_array, error = extract_face(image_cv2)
        if error:
            return jsonify({"message": error}), 400

        # Dự đoán
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions)) * 100
        predicted_index = int(np.argmax(predictions))

        if confidence < 10:
            predicted_name = "Unknown"
        else:
            predicted_name = class_labels[predicted_index]

        return jsonify({
            "predicted_name": predicted_name,
            "confidence": f"{confidence:.2f}%",
            "message": "Nhận diện thành công!" if predicted_name != "Unknown" else "Không đủ tự tin để xác định người."
        }), 200

    except Exception as e:
        return jsonify({
            "message": f"Có lỗi xảy ra trong quá trình xử lý: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
