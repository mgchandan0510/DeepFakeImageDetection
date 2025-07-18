import os
import cv2
import numpy as np

def load_images_from_subfolders(base_folder, image_size=(128, 128), limit_per_class=None):
    images = []
    labels = []

    classes = {'real': 0, 'fake': 1}

    for label_name, label in classes.items():
        folder_path = os.path.join(base_folder, label_name)
        count = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                img = img / 255.0  # normalize
                images.append(img)
                labels.append(label)
                count += 1
                if limit_per_class and count >= limit_per_class:
                    break

    return np.array(images), np.array(labels)
