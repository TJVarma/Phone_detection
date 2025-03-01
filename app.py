# app.py
import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, request
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Parameters
IMG_SIZE = 64
categories = ["iphone", "samsung", "motorola"]
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))

# Disable GPU (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load Model
MODEL_PATH = os.getenv("MODEL_PATH", "phone_brand_classifier.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure it is uploaded correctly.")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 for default webcam

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame as a response to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    port = int(os.getenv("PORT", 10000))  # Render's dynamic port
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
