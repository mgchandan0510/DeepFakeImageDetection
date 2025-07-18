
# 🧠 Deepfake Detection Web App (CNN + Flask)

This project is a **deepfake image detection** system using a **Convolutional Neural Network (CNN)** model built with Keras and served through a **Flask** web application.

## 🚀 Features

- Deep Learning model trained to classify images as **REAL** or **FAKE**
- Web interface to upload and predict images
- Easy-to-understand confidence score
- Reproducible pipeline from training to deployment

---

## 📁 Project Structure

```
├── app.py                  # Flask web server
├── model.py                # CNN model architecture
├── train_model.py          # Training script
├── predict_image.py        # Image prediction from CLI
├── utils.py                # Utility to load and preprocess data
├── deepfake_cnn_model.h5   # Trained model
├── templates/
│   └── index.html          # HTML frontend (not provided here)
└── static/uploads/         # Directory for uploaded images
```

---

## 🏗️ Model Architecture

The CNN consists of:
- 3 Convolution + MaxPooling layers
- 1 Fully connected Dense layer
- 1 Output layer with sigmoid activation (for binary classification)

Defined in [`model.py`](model.py) and compiled using:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## 🧠 How It Works

### 🔧 Train the Model

Make sure your dataset is structured as:
```
archive/real_vs_fake/real-vs-fake/
  ├── train/
  │   ├── real/
  │   └── fake/
  ├── valid/
  │   ├── real/
  │   └── fake/
  └── test/
      ├── real/
      └── fake/
```

Then run:
```bash
python train_model.py
```

This script:
- Loads and preprocesses images from the dataset
- Trains the CNN model
- Evaluates it on a test set
- Saves the trained model to `deepfake_cnn_model.h5`

---

### 🖼️ Predict from CLI

Use `predict_image.py`:
```bash
python predict_image.py
```
Edit the script to change the input image path.

---

### 🌐 Run the Web App

```bash
python app.py
```

Then visit `http://127.0.0.1:5000/` in your browser. Upload an image and get the prediction (REAL/FAKE) with confidence score.

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

<details>
<summary>📋 Click to see sample requirements.txt</summary>

```text
Flask
numpy
opencv-python
tensorflow
scikit-learn
```

</details>

---

## 📸 Sample Prediction Output

- **Input**: An image (128x128 resized)
- **Output**: "REAL Image" or "FAKE Image" with confidence

Example:
```
🧠 Predicted: FAKE Image
Confidence: 0.9456
```

---

## 📌 Credits

- Model trained using images from [real-vs-fake dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- Built with TensorFlow/Keras and Flask
