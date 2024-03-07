import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import base64
from flask_cors import CORS 

app = Flask(__name__)
CORS(app, origins='*')


model = load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return False
    elif classNo == 1:
        return True

def getResult(img_bytes):
    # Convert bytes to numpy array
    image = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((64, 64))
    image = np.array(image) / 255.0  # Normalize image
    input_img = np.expand_dims(image, axis=0)
    predictions = model.predict(input_img)
    predicted_classes = np.argmax(predictions, axis=-1)
    result = predicted_classes.item()
    return result

@app.route('/', methods=['GET'])
def index():
    return "Welcome to Brain Tumor Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    print("working")
    data = request.get_json()
    print(data['image'])
    if 'image' not in data:
        return jsonify({'error': 'No image provided'})
    
    img_bytes = base64.b64decode(data['image'])
    value = getResult(img_bytes)
    result = get_className(value)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)