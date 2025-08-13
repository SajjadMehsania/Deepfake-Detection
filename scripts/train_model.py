import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# check if GPU is available
if tf.config.list_physical_devices('GPU'):
    device = "/GPU:0"
else:
    device = "/CPU:0"
print("Using:", device)

# paths
train_dir = "../dataset/train_data"
val_dir = "../dataset/validation"
test_dir = "../dataset/test_data"

# make sure models folder exists
models_folder = "../models"
os.makedirs(models_folder, exist_ok=True)
model_path = os.path.join(models_folder, "best_hybrid_model.keras")

img_size = (224, 224)
batch_size = 16

# loading dataset
def process_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    return img, label

def get_dataset(folder, training=True):
    classes = sorted(os.listdir(folder))
    files = []
    labels = []
    for idx, c in enumerate(classes):
        c_path = os.path.join(folder, c)
        if os.path.isdir(c_path):
            for f in os.listdir(c_path):
                files.append(os.path.join(c_path, f))
                labels.append(idx)
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    ds = ds.map(process_image)
    if training:
        ds = ds.shuffle(100)
    return ds.batch(batch_size)

train_ds = get_dataset(train_dir, True)
val_ds = get_dataset(val_dir, False)

# test data
test_files = []
test_labels = []
test_classes = sorted(os.listdir(test_dir))
for idx, c in enumerate(test_classes):
    c_path = os.path.join(test_dir, c)
    for f in os.listdir(c_path):
        test_files.append(os.path.join(c_path, f))
        test_labels.append(idx)
test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
test_ds = test_ds.map(process_image).batch(batch_size)

# making hybrid model
input_img = Input(shape=(224, 224, 3))
mnet = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_img)
dnet = DenseNet121(weights="imagenet", include_top=False, input_tensor=input_img)

mnet.trainable = False
dnet.trainable = False

mnet_out = GlobalAveragePooling2D()(mnet.output)
dnet_out = GlobalAveragePooling2D()(dnet.output)

merge = Concatenate()([mnet_out, dnet_out])
x = BatchNormalization()(merge)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_img, outputs=output)
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
]

# training part 1
print("Training - phase 1")
with tf.device(device):
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# fine-tuning
mnet.trainable = True
dnet.trainable = True
for layer in mnet.layers[:-20]:
    layer.trainable = False
for layer in dnet.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])

print("Training - phase 2 (fine-tune)")
with tf.device(device):
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

# evaluation
model.load_weights(model_path)
y_pred_prob = model.predict(test_ds)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

acc = accuracy_score(test_labels, y_pred)
prec = precision_score(test_labels, y_pred)
rec = recall_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)
auc = roc_auc_score(test_labels, y_pred_prob)

print(classification_report(test_labels, y_pred, target_names=test_classes))
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("AUC:", auc)

# plot results
fpr, tpr, _ = roc_curve(test_labels, y_pred_prob)
cm = confusion_matrix(test_labels, y_pred)

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_classes, yticklabels=test_classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
