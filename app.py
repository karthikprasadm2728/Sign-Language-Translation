from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import base64
import os

app = Flask(__name__)

# Initialize components with error handling
try:
    # Load the trained model
    MODEL_PATH = os.path.join("Model", "mobilenetv3_sign_language_model.keras")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    
    # Labels for predictions
    labels = ["Hello", "Yes", "No", "I Love You", "Okay", "Please", "Thank You"]
    
    # Initialize hand detector
    detector = HandDetector(maxHands=1)
    
    # Image parameters
    offset = 20
    imgSize = 224
    
    print("✅ All components loaded successfully")
except Exception as e:
    print(f"❌ Initialization failed: {str(e)}")
    raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        img_base64 = data['image']
        img_data = img_base64.split(",")[1] if "," in img_base64 else img_base64
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Detect hand
        hands, _ = detector.findHands(img, draw=False)
        if not hands:
            return jsonify({'prediction': 'No hand detected'}), 200

        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure bounding box is within image bounds
        x1, y1 = max(x - offset, 0), max(y - offset, 0)
        x2, y2 = min(x + w + offset, img.shape[1]), min(y + h + offset, img.shape[0])
        imgCrop = img[y1:y2, x1:x2]
        
        if imgCrop.size == 0:
            return jsonify({'prediction': 'Invalid hand crop'}), 200

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
        
        return jsonify({
            'prediction': label,
            'confidence': float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({
            'error': 'Error processing image',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
