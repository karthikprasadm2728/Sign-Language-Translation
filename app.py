from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import base64
import os

# Load the trained model once
MODEL_PATH = "Model/mobilenetv3_sign_language_model.keras"
model = load_model(MODEL_PATH)

# Labels for predictions
labels = ["Hello", "Yes", "No", "I Love You", "Okay", "Please", "Thank You"]

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Image parameters
offset = 20
imgSize = 224

# Flask setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_base64 = data.get('image')

    if not img_base64:
        return jsonify({'prediction': 'No image provided'})

    try:
        # Decode image
        img_data = img_base64.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect hand
        hands, _ = detector.findHands(img, draw=False)

        if not hands:
            return jsonify({'prediction': 'No hand detected'})

        hand = hands[0]
        x, y, w, h = hand['bbox']
        x1, y1 = max(x - offset, 0), max(y - offset, 0)
        x2, y2 = min(x + w + offset, img.shape[1]), min(y + h + offset, img.shape[0])
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            return jsonify({'prediction': 'Invalid hand crop'})

        # Resize and center hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Prepare image for prediction
        imgInput = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        imgInput = np.expand_dims(imgInput, axis=0).astype(np.float32) / 255.0

        # Predict
        prediction = model.predict(imgInput)
        label = labels[np.argmax(prediction)]

        return jsonify({'prediction': label})

    except Exception as e:
        return jsonify({'prediction': 'Error processing image', 'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
