
from flask import Flask, request, jsonify, render_template, session, url_for, redirect
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename
import pickle
from PIL import Image
import uuid

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'plantcare-ai-secret-key'  # Required for session
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and class names
print("Loading model...")
model = load_model('model/mobilenetv2_best.keras')
print("✅ Model loaded successfully")

with open('model/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)
print(f"✅ Class names loaded: {len(class_names)} classes")

def predict_image(img_path):
    """Predict disease from image path"""
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get class name
    disease_name = class_names[predicted_class_idx]
    
    return {
        'disease': disease_name,
        'confidence': confidence,
        'confidence_percentage': round(confidence * 100, 2)
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Generate unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make prediction
            result = predict_image(filepath)
            
            # Save to session for results page
            session['prediction'] = result
            session['image_path'] = filepath
            
            return jsonify({'success': True, 'redirect': '/result'})
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    """Results page"""
    prediction = session.get('prediction')
    image_path = session.get('image_path')
    
    if not prediction or not image_path:
        return redirect(url_for('upload_page'))
    
    # Make path relative for template
    image_relative = image_path.replace('static/', '')
    
    return render_template('result.html', 
                          prediction=prediction, 
                          image_path=image_relative)

@app.route('/api/classes')
def get_classes():
    """API endpoint to get all class names"""
    return jsonify({'classes': class_names})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'classes': len(class_names)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
