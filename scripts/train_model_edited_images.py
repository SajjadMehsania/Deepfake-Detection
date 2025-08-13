import os
import time
import shutil
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve

# runtime
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# config
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 8
EPOCHS_PHASE2 = 8
LEARNING_RATE_PHASE1 = 1e-4
LEARNING_RATE_PHASE2 = 5e-6
UNFREEZE_LAST_LAYERS = 10

TRAIN_DIR = r"C:\Users\sajja\Desktop\Deepfake-Detection\dataset\realvsedited\train_data"
VAL_DIR   = r"C:\Users\sajja\Desktop\Deepfake-Detection\dataset\realvsedited\validation"
TEST_DIR  = r"C:\Users\sajja\Desktop\Deepfake-Detection\dataset\realvsedited\test_data"

CACHE_ROOT  = r"C:\Users\sajja\Desktop\Deepfake-Detection\dataset_ela_cache"
TRAIN_CACHE = os.path.join(CACHE_ROOT, "train_data")
VAL_CACHE   = os.path.join(CACHE_ROOT, "validation")
TEST_CACHE  = os.path.join(CACHE_ROOT, "test_data")

MODEL_DIR = os.path.join("..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "hybrid_ela_mobilenetv2_densenet121.keras")
BEST_WEIGHTS = os.path.join(MODEL_DIR, "best_weights.keras")

# ELA utils
def perform_ela_to_path(src_path, dst_path, quality=90):
    orig = cv2.imread(src_path)
    if orig is None:
        black = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
        cv2.imwrite(dst_path, black)
        return
    tmp_jpg = dst_path + ".tmp.jpg"
    cv2.imwrite(tmp_jpg, orig, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    compressed = cv2.imread(tmp_jpg)
    if compressed is None:
        black = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
        cv2.imwrite(dst_path, black)
        try:
            os.remove(tmp_jpg)
        except OSError:
            pass
        return
    diff = cv2.absdiff(orig, compressed)
    max_diff = np.max(diff)
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    ela_img = (diff * scale).astype(np.uint8)
    ela_img = cv2.resize(ela_img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst_path, ela_img)
    try:
        os.remove(tmp_jpg)
    except OSError:
        pass

def build_ela_cache_split(src_dir, dst_dir, num_workers=8, overwrite=False):
    if overwrite and os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    class_names = sorted(d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d)))
    for cls in class_names:
        os.makedirs(os.path.join(dst_dir, cls), exist_ok=True)
    tasks = []
    for cls in class_names:
        src_cls = os.path.join(src_dir, cls)
        dst_cls = os.path.join(dst_dir, cls)
        for fname in os.listdir(src_cls):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                src_path = os.path.join(src_cls, fname)
                out_name = os.path.splitext(fname)[0] + "_ela.jpg"
                dst_path = os.path.join(dst_cls, out_name)
                if not overwrite and os.path.exists(dst_path):
                    continue
                tasks.append((src_path, dst_path))
    if not tasks:
        return
    print(f"Building ELA cache at: {dst_dir} ({len(tasks)} files)...")
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(perform_ela_to_path, s, d) for s, d in tasks]
        done = 0
        for _ in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  Cached {done}/{len(tasks)} images...")

def ensure_cache(overwrite=False):
    build_ela_cache_split(TRAIN_DIR, TRAIN_CACHE, num_workers=8, overwrite=overwrite)
    build_ela_cache_split(VAL_DIR,   VAL_CACHE,   num_workers=6, overwrite=overwrite)
    build_ela_cache_split(TEST_DIR,  TEST_CACHE,  num_workers=6, overwrite=overwrite)
    print("ELA cache ready.")

# data pipeline
def build_dataset_from_dir(root_dir, batch_size, shuffle=True, augment=False, seed=42):
    class_names = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    file_paths, labels = [], []
    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(".jpg"):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])
    file_paths = np.array(file_paths)
    labels = np.array(labels, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=seed, reshuffle_each_iteration=True)
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.05)
            img = tf.image.random_contrast(img, 0.95, 1.05)
        return img, tf.cast(label, tf.int32)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, class_names, labels

# model
def build_hybrid_model():
    inputs = Input(shape=(*IMAGE_SIZE, 3))
    mobilenet = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    densenet = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=inputs)
    mobilenet.trainable = False
    densenet.trainable = False
    gap_mobilenet = GlobalAveragePooling2D(name="gap_mobilenet")(mobilenet.output)
    gap_densenet  = GlobalAveragePooling2D(name="gap_densenet")(densenet.output)
    features = Concatenate(name="concat_features")([gap_mobilenet, gap_densenet])
    x = BatchNormalization()(features)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', name="pred")(x)
    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_MobileNetV2_DenseNet121")
    return model, mobilenet, densenet

# train/eval
if __name__ == "__main__":
    ensure_cache(overwrite=False)

    train_ds, train_classes, train_labels = build_dataset_from_dir(TRAIN_CACHE, BATCH_SIZE, shuffle=True, augment=True)
    val_ds,   val_classes,   val_labels   = build_dataset_from_dir(VAL_CACHE,   BATCH_SIZE, shuffle=False, augment=False)
    test_ds,  test_classes,  test_labels  = build_dataset_from_dir(TEST_CACHE,  BATCH_SIZE, shuffle=False, augment=False)

    print(f"Classes: {train_classes}")
    print(f"Training samples: {len(train_labels)}, Validation samples: {len(val_labels)}, Testing samples: {len(test_labels)}")

    cls_ids, cls_counts = np.unique(train_labels, return_counts=True)
    total = cls_counts.sum()
    class_weights = {int(c): float(total / (len(cls_ids) * cnt)) for c, cnt in zip(cls_ids, cls_counts)}
    print(f"Class weights: {class_weights}")

    model, mobilenet, densenet = build_hybrid_model()
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_PHASE1), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(BEST_WEIGHTS, monitor='val_loss', save_best_only=True),
    ]

    print("\nPhase 1: Training classification head...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE1,
        class_weight=class_weights,
        verbose=1,
        callbacks=callbacks
    )

    print(f"\nUnfreezing last {UNFREEZE_LAST_LAYERS} layers of base models for fine-tuning...")
    for backbone in (mobilenet, densenet):
        for layer in backbone.layers[-UNFREEZE_LAST_LAYERS:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_PHASE2), loss='binary_crossentropy', metrics=['accuracy'])

    print("\nPhase 2: Fine-tuning...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PHASE2,
        class_weight=class_weights,
        verbose=1,
        callbacks=callbacks
    )

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved at: {MODEL_SAVE_PATH}")

    if os.path.exists(BEST_WEIGHTS):
        model.load_weights(BEST_WEIGHTS)
        print(f"Loaded best weights from: {BEST_WEIGHTS}")

    print("\nEvaluating on test data...")
    y_test_pred_prob = model.predict(test_ds, verbose=1).flatten()

    y_val_pred_prob = model.predict(val_ds, verbose=1).flatten()
    precisions, recalls, thresholds = precision_recall_curve(val_labels, y_val_pred_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.nanargmax(f1_scores))
