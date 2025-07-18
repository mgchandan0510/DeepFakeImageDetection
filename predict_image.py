import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model("deepfake_cnn_model.h5")  # or .keras if you used the newer format

# Load and preprocess the image
img_path = "00AUP94LQS.jpg"  # 🧠 Replace this with your image path
img = image.load_img(img_path, target_size=(128, 128))  # Must match training size
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)[0][0]

# Result
if prediction > 0.5:
    print("🧠 Predicted: FAKE Image")
else:
    print("🧠 Predicted: REAL Image")

print(f"Confidence: {prediction:.4f}")
