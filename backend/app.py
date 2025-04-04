import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import os


# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)  # Thêm CORS để cho phép truy cập từ các domain khác

# Tải mô hình EfficientNetB0 đã huấn luyện
model = tf.keras.models.load_model('model/model/restnet50.keras')

# Danh sách các lớp (class names) tương ứng với mô hình
class_names = ['Hùng Anh', 'Ngọc Anh', 'Tiểu Long', 'Nhật Tùng', 'Thành Lộc', 'Minh Long']  # Bạn thay thế với các lớp thực tế

# Chuyển đổi ảnh thành định dạng mà mô hình có thể nhận dạng
def prepare_image(image_file):
    # Đọc ảnh từ file
    image = Image.open(image_file)
    
    # Chuyển ảnh về dạng numpy array và chuẩn hóa kích thước
    image = image.convert('RGB')
    image = image.resize((224, 224))  # Đảm bảo kích thước khớp với mô hình của bạn
    image_array = np.array(image)
    
    # Nếu mô hình yêu cầu chuẩn hóa ảnh, ví dụ:
    image_array = image_array / 255.0  # Chuẩn hóa ảnh về [0, 1]
    
    # Thêm một chiều batch vào ảnh (chuyển từ (224, 224, 3) thành (1, 224, 224, 3))
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.route('/face-detect', methods=['POST'])
def detect_face():

    if 'image' not in request.files:
        return jsonify({"message": "Không có ảnh trong yêu cầu!"}), 400
    
    # Lấy ảnh từ request
    file = request.files['image']
    
    if file:
        try:
            # Chuyển ảnh từ request thành định dạng có thể xử lý được
            image_array = prepare_image(file)
            
            # Dự đoán với mô hình
            predictions = model.predict(image_array)
            
            # Giả sử mô hình trả về các xác suất cho từng lớp
            predicted_class = np.argmax(predictions)  # Lớp có xác suất cao nhất
            
            # Dùng tên lớp cho dự đoán
            predicted_name = class_names[predicted_class]
            
            # Trả về kết quả nhận diện khuôn mặt
            return jsonify({
                'predicted_class': int(predicted_class),
                'predicted_name': predicted_name,
                'message': 'Nhận diện thành công!'
            }), 200
        
        except Exception as e:
            return jsonify({
                'message': f'Có lỗi trong quá trình xử lý ảnh: {str(e)}'
            }), 500

    else:
        return jsonify({"message": "Không nhận được ảnh!"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
