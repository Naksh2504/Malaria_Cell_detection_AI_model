import os
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import traceback

app = Flask(__name__, static_folder='.')
CORS(app)

IMG_SIZE = (128, 128)
MODEL_PATH = 'malaria_model.h5'

def create_default_model():
    """Builds and compiles default MobileNetV2 architecture if model file is missing."""
    print("Building default MobileNetV2 architecture...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    m = tf.keras.Model(inputs, outputs)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.save(MODEL_PATH)
    print(f"Default model saved to {MODEL_PATH}")
    return m

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model successfully loaded from {MODEL_PATH}")
    else:
        model = create_default_model()
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    try:
        model = create_default_model()
    except Exception as fallback_err:
        print(f"Error creating fallback model: {fallback_err}")
        model = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/samples/<path:filename>')
def serve_sample(filename):
    return send_from_directory('samples', filename)

def preprocess_image(image_bytes):
    """
    Takes image bytes, opens the image, resizes it, and prepares it
    for the model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    return img_array
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded!'}), 500        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        image_bytes = file.read()
        preprocessed_image = preprocess_image(image_bytes)
        prediction = model.predict(preprocessed_image)
        confidence = prediction[0][0]

        if confidence < 0.5:
            label = 'Parasitized'
            confidence_percent = (1 - confidence) * 100
        else:
            label = 'Uninfected'
            confidence_percent = confidence * 100
        return jsonify({
            'prediction': label,
            'confidence': float(confidence_percent) 
        })
    except Exception as e:
        print("\n Error during prediction")
        traceback.print_exc()
        return jsonify({'error': f'An error occurred on the server: {str(e)}'}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
