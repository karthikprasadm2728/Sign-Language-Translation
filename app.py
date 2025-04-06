from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import os
from PIL import Image
import io
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Load model
MODEL_PATH = os.path.join("model", "sign_language_model.h5")  # Use .h5 format for smaller size
model = load_model(MODEL_PATH)

# Labels
LABELS = ["Hello", "Yes", "No", "I Love You", "Okay", "Please", "Thank You"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        data = request.get_json()
        img_data = data['image'].split(",")[1] if "," in data['image'] else data['image']
        img_bytes = base64.b64decode(img_data)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img)
        
        # Hand detection with MediaPipe
        results = hands.process(img_array)
        if not results.multi_hand_landmarks:
            return jsonify({'prediction': 'No hand detected'})
        
        # Get hand bounding box
        landmarks = results.multi_hand_landmarks[0].landmark
        x_coords = [landmark.x * img.width for landmark in landmarks]
        y_coords = [landmark.y * img.height for landmark in landmarks]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Crop and resize
        crop = img.crop((x_min, y_min, x_max, y_max))
        crop = crop.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(crop) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        label = LABELS[np.argmax(prediction)]
        
        return jsonify({
            'prediction': label,
            'confidence': float(np.max(prediction))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
