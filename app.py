from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)
model = load_model('model.keras')

# Define a mapping from class indices to class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load and preprocess the image
        img = image.load_img(io.BytesIO(file.read()))  # Read and load the image from the file object
        img_array = preprocess_image(img)
        
        print("Preprocessed image array shape:", img_array.shape)  # Debugging line
        
        # Predict
        predictions = model.predict(img_array)
        
        print("Model predictions:", predictions)  # Debugging line
        
        # Reshape predictions if necessary and extract class index
        predictions = np.squeeze(predictions)  # Remove extra dimensions if present
        
        print("Reshaped predictions:", predictions)  # Debugging line
        
        predicted_index = np.argmax(predictions)  # Get the index of the max value
        predicted_class_name = class_names[predicted_index]  # Map index to class name
        
        print("Predicted class name:", predicted_class_name)  # Debugging line
        
        return jsonify({'predicted_class': predicted_class_name})
    except Exception as e:
        print("Error during prediction:", str(e))  # Debugging line
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
