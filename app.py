from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model("deepfake_cnn_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    result = "FAKE Image" if prediction > 0.5 else "REAL Image"
    confidence = f"{prediction:.4f}"

    return render_template('index.html', result=result, confidence=confidence, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
