import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, 'backend', 'model', 'ResNet50V2_model_t2.keras')
file_path = MODEL_PATH

if os.path.exists(file_path):
    print(f"File {file_path} exists.")
else:
    print(f"File {file_path} does not exist.")


# Đường dẫn tới mô hình .keras hiện tại
keras_model_path = r'D:/Work/CV/Smart-Attendance-AI/backend/model.keras'

# Đường dẫn lưu mô hình mới dưới định dạng .h5
h5_model_path = '../backend/model.h5'

def convert_model(keras_model_path, h5_model_path):
    try:
        # Tải mô hình .keras
        model = load_model(keras_model_path)
        print(f"Model {keras_model_path} loaded successfully.")

        # Lưu lại mô hình dưới dạng .h5
        model.save(h5_model_path)
        print(f"Model saved as {h5_model_path}.")
    except Exception as e:
        print(f"Error during model conversion: {e}")

if __name__ == "__main__":
    convert_model(keras_model_path, h5_model_path)
