from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import base64
import os

# Load the trained model
model_path = "Model/mobilenetv3_sign_language_model.keras"
model = load_model(model_path)

# Labels for sign language gestures
labels = ["Hello", "Yes", "No", "I Love You", "Okay", "Please", "Thank You"]

# Hand Detector
detector = HandDetector(maxHands=1)

# Image Processing Parameters
offset = 20
imgSize = 224

# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'prediction': 'No image provided'})

    try:
        # Decode base64 image
        img_data = data['image'].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        hands, img = detector.findHands(img, draw=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            x1, y1 = max(x - offset, 0), max(y - offset, 0)
            x2, y2 = min(x + w + offset, img.shape[1]), min(y + h + offset, img.shape[0])
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                return jsonify({'prediction': "No hand"})

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

            imgInput = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
            imgInput = np.expand_dims(imgInput, axis=0).astype(np.float32) / 255.0
            prediction = model.predict(imgInput)
            label = labels[np.argmax(prediction)]
            return jsonify({'prediction': label})

        return jsonify({'prediction': 'No hand detected'})

    except Exception as e:
        return jsonify({'prediction': 'Error processing image', 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
