# app.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response

@app.route('/')
def index():
    return render_template('index.html')  # No 'templates/' prefix

# Parameters
IMG_SIZE = 64
categories = ["Google Pixel","Iphone","Samsung"]
CONFIDENCE_THRESHOLD = 0.5  # Confidence below this = "No Phone"

# Load the trained model
model = load_model("phone_brand_classifier_2.h5")

# Flask app
app = Flask(__name__)

# Webcam setup
cap = cv2.VideoCapture(0)  # 0 is default webcam
if not cap.isOpened():
    raise Exception("Error: Could not open webcam.")

# Prediction function
def predict_phone(frame):
    img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    prediction = model.predict(img_input, verbose=0)  # Silent prediction
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    
    if confidence < CONFIDENCE_THRESHOLD:
        return "No Phone", confidence
    return categories[class_idx], confidence

# Generate video frames with predictions
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict phone brand
        brand, confidence = predict_phone(frame)
        label = f"{brand} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Cleanup on shutdown
def shutdown():
    cap.release()

import atexit
atexit.register(shutdown)

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)