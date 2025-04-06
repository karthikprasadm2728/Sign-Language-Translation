from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'Model/mobilenetv3_sign_language_model.keras')
model = tf.keras.models.load_model(model_path)

# Classes (update based on your training)
CLASSES = ['Hello', 'Yes', 'No', 'I Love You', 'Okay', 'Please', 'Thank You']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']  # base64 image string

    # Decode base64
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))  # Resize as per your model input
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    result = CLASSES[class_index]

    return {'prediction': result}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
