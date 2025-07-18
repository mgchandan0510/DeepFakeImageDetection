from utils import load_images_from_subfolders
from model import build_cnn_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# 1. Load Data from Train, Valid, and Test folders
print("Loading datasets...")

X_train_all, y_train_all = load_images_from_subfolders(
    base_folder='archive/real_vs_fake/real-vs-fake/train',
    image_size=(128, 128),
    limit_per_class=2000
)

X_valid, y_valid = load_images_from_subfolders(
    base_folder='archive/real_vs_fake/real-vs-fake/valid',
    image_size=(128, 128),
    limit_per_class=1000
)

X_test, y_test = load_images_from_subfolders(
    base_folder='archive/real_vs_fake/real-vs-fake/test',
    image_size=(128, 128),
    limit_per_class=1000
)

# 2. Combine valid with train split (optional)
print("Combining and splitting data...")
X_train_combined = np.concatenate((X_train_all, X_valid), axis=0)
y_train_combined = np.concatenate((y_train_all, y_valid), axis=0)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_combined, y_train_combined, test_size=0.2, random_state=42, stratify=y_train_combined
)

# 3. Build CNN Model
print("Building model...")
model = build_cnn_model(input_shape=(128, 128, 3))

# 4. Train the Model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# 5. Evaluate on test set
print("Evaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f" Test Accuracy: {test_acc * 100:.2f}%")

# 6. Save the Model
print("Saving model...")
model.save("deepfake_cnn_model.h5")
print(" Training complete. Model saved as deepfake_cnn_model.h5")
