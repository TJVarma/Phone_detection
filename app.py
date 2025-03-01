# app.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response, request
import os

# Initialize Flask app first
app = Flask(__name__)

# Parameters
IMG_SIZE = 64
categories = ["iphone", "samsung", "motorola"]
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
MODEL_PATH = os.getenv("MODEL_PATH", "/app/phone_brand_classifier.h5")  # Render path
model = load_model(MODEL_PATH)

def predict_phone(image):
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    prediction = model.predict(img_input, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    if confidence < CONFIDENCE_THRESHOLD:
        return "No Phone", confidence
    return categories[class_idx], confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            brand, confidence = predict_phone(img)
            return render_template('index.html', prediction=f"{brand} ({confidence:.2f})")
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Render's dynamic port
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)