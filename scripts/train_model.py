import os
import numpy as np
import tensorflow as tf

# --- 1. SETUP AND CONFIGURATION ---
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Using device: {device}")

# --- 2. PARAMETERS ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16   # Lower batch size for limited memory
AUTOTUNE = tf.data.AUTOTUNE

# FOLDER PATHS
train_dir = os.path.join("..", "dataset", "train_data")
val_dir   = os.path.join("..", "dataset", "validation")
test_dir  = os.path.join("..", "dataset", "test_data")
MODEL_SAVE_PATH = os.path.join("..", "models", "best_hybrid_model.keras")

# --- 3. DATA PIPELINE ---
def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img, label

def create_dataset(data_dir, is_training=True):
    ds = tf.data.Dataset.list_files(os.path.join(data_dir, '/'), shuffle=is_training)
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2] == class_names[1]

    labeled_ds = ds.map(lambda x: (x, get_label(x)), num_parallel_calls=AUTOTUNE)
    processed_ds = labeled_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    # Disable caching for large datasets on limited RAM
    # if is_training:
    #     processed_ds = processed_ds.cache()
    
    return processed_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Create datasets
train_ds = create_dataset(train_dir, is_training=True)
val_ds = create_dataset(val_dir, is_training=False)

# Prepare test dataset
test_image_paths = sorted(tf.io.gfile.glob(os.path.join(test_dir, '/')))
test_class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
y_true = [1 if os.path.basename(os.path.dirname(p)) == test_class_names[1] else 0 for p in test_image_paths]
test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_ds = test_path_ds.map(lambda x: process_path(x, 0)[0]).batch(BATCH_SIZE)

# --- 4. HYBRID MODEL DEFINITION ---
input_tensor = Input(shape=(*IMAGE_SIZE, 3))

base_mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_tensor)
base_densenet = DenseNet121(weights="imagenet", include_top=False, input_tensor=input_tensor)

base_mobilenet.trainable = False
base_densenet.trainable = False

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
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)

# --- 5. TRAINING & FINE-TUNING ---
print("\n--- Training Phase 1: Transfer Learning ---\n")
with tf.device(device):
    model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[early_stop, checkpoint, lr_scheduler]
    )

base_mobilenet.trainable = True
base_densenet.trainable = True

for layer in base_mobilenet.layers[:-20]:
    layer.trainable = False
for layer in base_densenet.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("\n--- Training Phase 2: Fine-Tuning ---\n")
with tf.device(device):
    model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=[early_stop, checkpoint, lr_scheduler]
    )

# --- 6. EVALUATION ---
print("\n--- Evaluating on test data... ---\n")
model.load_weights(MODEL_SAVE_PATH)
y_pred_prob = model.predict(test_ds, verbose=1)
y_pred = (y_pred_prob >= 0.5).astype('int32').flatten()

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)

print("\n--- CLASSIFICATION REPORT ---\n")
print(classification_report(y_true, y_pred, target_names=test_class_names))
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC AUC    : {auc:.4f}")

# VISUALIZATIONS
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_class_names, yticklabels=test_class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()