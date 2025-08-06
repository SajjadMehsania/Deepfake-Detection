import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model

# 1. Set parameters
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join("..", "models", "best_hybrid_model.keras")
TEST_IMAGE_FOLDER = r"C:\Users\sajja\Desktop\Deepfake-Detection\dataset\test_images"

# 2. Define preprocessing
def preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img

def load_images_from_folder(folder):
    exts = (".jpg", ".jpeg", ".png")
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    image_paths.sort()  
    images = [preprocess_image(p) for p in image_paths]
    images = tf.stack(images)
    return images, image_paths

# 3. (Re)Build Model Structure
def build_model():
    input_tensor = Input(shape=(*IMAGE_SIZE, 3))
    base_mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)
    base_densenet = DenseNet121(weights="imagenet", include_top=False, input_tensor=input_tensor)
    base_mobilenet.trainable = True
    base_densenet.trainable = True
    features_mobilenet = base_mobilenet.output
    features_mobilenet = GlobalAveragePooling2D()(features_mobilenet)
    features_densenet = base_densenet.output
    features_densenet = GlobalAveragePooling2D()(features_densenet)
    concatenated_features = Concatenate()([features_mobilenet, features_densenet])
    x = BatchNormalization()(concatenated_features)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=output)
    return model

# 4. Run prediction
if __name__ == '__main__':
    print("Loading images from:", TEST_IMAGE_FOLDER)
    images, image_paths = load_images_from_folder(TEST_IMAGE_FOLDER)
    print(f"Found {len(image_paths)} images.")
    
    print("Rebuilding and loading model...")
    model = build_model()
    model.load_weights(MODEL_PATH)
    print("Model loaded.")

    print("Predicting...")
    preds = model.predict(images, batch_size=4)
    preds_cls = (preds >= 0.5).astype(int).flatten()

    # Print results
    print("\nResults:")
    print("Image\t\t\t\tPrediction\tProbability")
    for path, pred, prob in zip(image_paths, preds_cls, preds):
        label = "real" if pred == 1 else "fake"
        print(f"{os.path.basename(path):30s}\t{label:5s}\t\t{prob[0]:.4f}")

    # Optionally, save results to file
    # with open("my_test_results.csv", "w") as f:
    #     f.write("image,prediction,probability\n")
    #     for path, pred, prob in zip(image_paths, preds_cls, preds):
    #         label = "real" if pred == 1 else "fake"
    #         f.write(f"{os.path.basename(path)},{label},{prob[0]:.4f}\n")
