from flask import Flask, render_template, Response
import cv2
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Load the trained model
model_path = "Model/mobilenetv3_sign_language_model.keras"
model = load_model(model_path)

# Labels for sign language gestures
labels = ["Hello", "Yes", "No", "I Love You", "Okay", "Please", "Thank You"]

# Initialize Camera & Hand Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Image Processing Parameters
offset = 20
imgSize = 224  # MobileNetV3 requires 224x224 input size

# Flask Application
app = Flask(__name__)

# Function to capture video feed and send it to the front-end
def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img, draw=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure valid cropping dimensions
            x1, y1 = max(x - offset, 0), max(y - offset, 0)
            x2, y2 = min(x + w + offset, img.shape[1]), min(y + h + offset, img.shape[0])

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Resize Keeping Aspect Ratio
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize), interpolation=cv2.INTER_AREA)
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal), interpolation=cv2.INTER_AREA)
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Convert Image for Model
            imgInput = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
            imgInput = np.expand_dims(imgInput, axis=0)  # Add batch dimension
            imgInput = imgInput.astype(np.float32) / 255.0  # Normalize

            # Predict with the model
            prediction = model.predict(imgInput)

            if prediction.size > 0:
                index = np.argmax(prediction)  # Get predicted label index
                label = labels[index]
            else:
                label = "Unknown"

            # Add label to the frame
            cv2.putText(imgOutput, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame in JPEG format and send it to the front-end
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Renders the UI

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render will set PORT; fallback to 5000 for local dev
    app.run(debug=True, host='0.0.0.0', port=port)

