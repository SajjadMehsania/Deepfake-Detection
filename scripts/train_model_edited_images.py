import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# Constants and Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 15
LEARNING_RATE_PHASE1 = 1e-4
LEARNING_RATE_PHASE2 = 1e-5
VALIDATION_SPLIT = 0.2

# Dataset directories - adjust if needed
TRAIN_DIR = r"C:\Users\sajja\Desktop\Deepfake-Detection\dataset\realvsedited\training"
TEST_DIR = r"C:\Users\sajja\Desktop\Deepfake-Detection\dataset\realvsedited\testing"

# ELA preprocessing helper function
def perform_ela(image_path, quality=90):
    orig = cv2.imread(image_path)
    temp_path = "temp_ela.jpg"
    cv2.imwrite(temp_path, orig, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    compressed = cv2.imread(temp_path)
    diff = cv2.absdiff(orig, compressed)
    max_diff = np.max(diff)
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_img = (diff * scale).astype(np.uint8)
    ela_img = cv2.resize(ela_img, IMAGE_SIZE)
    return ela_img

# Custom data generator with on-the-fly ELA preprocessing
class ELAImageGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.image_paths[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]
        images = np.array([perform_ela(p) for p in batch_paths], dtype=np.float32) / 255.0
        labels = np.array(batch_labels)
        return images, labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Utility to collect image file paths and labels given folder with class subfolders
def get_image_paths_and_labels(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    image_paths = []
    labels = []
    for cls in classes:
        cls_folder = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(cls_folder, fname))
                labels.append(class_map[cls])
    return image_paths, labels, class_map

# Build combined DenseNet121 + MobileNetV2 model
def build_model():
    input_tensor = Input(shape=(*IMAGE_SIZE, 3))

    mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
    densenet = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)

    mobilenet.trainable = False
    densenet.trainable = False

    features_mobilenet = GlobalAveragePooling2D()(mobilenet.output)
    features_densenet = GlobalAveragePooling2D()(densenet.output)

    concatenated = Concatenate()([features_mobilenet, features_densenet])

    x = BatchNormalization()(concatenated)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=output)

    return model, mobilenet, densenet

if __name__ == "__main__":
    print("Loading training data paths and labels...")
    train_paths, train_labels, class_map_train = get_image_paths_and_labels(TRAIN_DIR)
    print(f"Classes found: {class_map_train}")
    
    # Split into train and validation sets preserving class distribution
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=VALIDATION_SPLIT, stratify=train_labels, random_state=42)
    print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    print("Loading test data paths and labels...")
    test_paths, test_labels, class_map_test = get_image_paths_and_labels(TEST_DIR)
    print(f"Classes found: {class_map_test}")

    # Instantiate data generators
    train_gen = ELAImageGenerator(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = ELAImageGenerator(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)
    test_gen = ELAImageGenerator(test_paths, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    # Compute balanced class weights to handle imbalance
    class_weights_arr = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights_arr))
    print(f"Class weights: {class_weights}")

    # Build and compile model
    model, mobilenet, densenet = build_model()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1),
                  loss='binary_crossentropy', metrics=['accuracy'])

    print("\nStarting Phase 1 training (head only)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        class_weight=class_weights,
        verbose=2
    )

    # Unfreeze last 20 layers of each backbone model for fine-tuning
    print("\nUnfreezing last 20 layers of MobileNetV2 and DenseNet121...")
    for layer in mobilenet.layers[-20:]:
        layer.trainable = True
    for layer in densenet.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2),
                  loss='binary_crossentropy', metrics=['accuracy'])

    print("\nStarting Phase 2 fine-tuning...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        class_weight=class_weights,
        verbose=2
    )

    # Save the final model
    model_save_path = os.path.join("..", "models", "combined_ela_densenet_mobilenetv2_detector.keras")
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    # Predict probabilities on validation to find best threshold
    y_val_pred_prob = model.predict(val_gen).flatten()
    precisions, recalls, thresholds = precision_recall_curve(val_labels, y_val_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"\nBest threshold on validation set: {best_threshold:.3f} (F1: {f1_scores[best_idx]:.4f})")

    # Predict and evaluate on test set with best threshold
    y_test_pred_prob = model.predict(test_gen).flatten()
    y_test_pred = (y_test_pred_prob >= best_threshold).astype(int)

    print("\nClassification Report on Test Data:")
    print(classification_report(test_labels, y_test_pred, target_names=[cls for cls in sorted(class_map_test.keys())]))

    print(f"Accuracy: {accuracy_score(test_labels, y_test_pred):.4f}")
    print(f"Precision: {precision_score(test_labels, y_test_pred):.4f}")
    print(f"Recall: {recall_score(test_labels, y_test_pred):.4f}")
    print(f"F1 Score: {f1_score(test_labels, y_test_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(test_labels, y_test_pred_prob):.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_labels, y_test_pred_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(test_labels, y_test_pred_prob):.4f})', color='darkorange')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Edited Image Detection')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
