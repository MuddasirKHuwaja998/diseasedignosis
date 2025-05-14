import os
import random
import json
import base64
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local model file path
MODEL_PATH = os.path.join(BASE_DIR, "model_storage", "disease_model_v1.h5")

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure the file exists in the correct path.")

# Load the model
model = load_model(MODEL_PATH)

# Define class names
class_names = [
    "Otite Media Acuta", "Otite Media Cronica", "Ventilazione dell'Orecchio", "Tappo di Cerume",
    "Corpo Estraneo", "Miringosclerosi", "Normale", "Otite Esterna",
    "Pseudomembranosa", "Timpanoclerosi"
]

# Load disease information
json_path = os.path.join(BASE_DIR, "static", "disease_info.json")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Disease information file not found at {json_path}. Ensure the file exists.")

with open(json_path, encoding='utf-8') as f:
    disease_info = json.load(f)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'tiff'}

def convert_to_jpeg(file_storage):
    """Convert TIFF or other formats to JPEG and return as BytesIO."""
    file_data = BytesIO(file_storage.read())
    img = Image.open(file_data)
    converted_img = BytesIO()

    # Convert image to RGB mode (required for JPEG) and save as JPEG
    img = img.convert("RGB")
    img.save(converted_img, format="JPEG")
    converted_img.seek(0)

    return converted_img

def predict_image(file_storage):
    """Predict the disease class for the given image."""
    jpeg_data = convert_to_jpeg(file_storage)
    img = image.load_img(jpeg_data, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction[0])

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle the upload and prediction logic."""
    if request.method == 'POST':
        left_file = request.files.get('left_ear')
        right_file = request.files.get('right_ear')

        if not left_file or not right_file:
            return render_template('index.html', error="Please upload both ear images.")

        if allowed_file(left_file.filename) and allowed_file(right_file.filename):
            left_index = predict_image(left_file)
            left_file.seek(0)
            right_index = predict_image(right_file)
            right_file.seek(0)

            left_disease = class_names[left_index]
            right_disease = class_names[right_index]

            left_info = random.choice(disease_info.get(left_disease, [{}]))
            right_info = random.choice(disease_info.get(right_disease, [{}]))

            left_jpeg = convert_to_jpeg(left_file)
            right_jpeg = convert_to_jpeg(right_file)
            left_image_data = base64.b64encode(left_jpeg.read()).decode('utf-8')
            right_image_data = base64.b64encode(right_jpeg.read()).decode('utf-8')

            results = {
                "left": {
                    "disease": left_disease,
                    "image_data": left_image_data,
                    "description": left_info.get('description', "Description not available."),
                    "causes": left_info.get('causes', ["Causes not available."]),
                    "cautions": left_info.get('cautions', ["Cautions not available."])
                },
                "right": {
                    "disease": right_disease,
                    "image_data": right_image_data,
                    "description": right_info.get('description', "Description not available."),
                    "causes": right_info.get('causes', ["Causes not available."]),
                    "cautions": right_info.get('cautions', ["Cautions not available."])
                }
            }
            return render_template('results.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
