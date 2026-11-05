"""
Inference Time Measurement
===========================
Measures per-image latency for:
  1. Preprocessing time  (CLAHE + resize)
  2. Feature extraction  (PCA_SVD_NMF_HOG)
  3. Classification time (MLP forward pass)
  4. Total end-to-end latency

Runs N_WARMUP warm-up passes then N_RUNS measurements.
Reports mean ± std in milliseconds.
"""

import os, cv2, numpy as np, time, warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout,
                                      BatchNormalization, Input)
from tensorflow.keras.optimizers import AdamW
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (EarlyStopping,
                                         LearningRateScheduler)
from imutils import paths
import math
warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================
IMAGE_SIZE = 64
N_WARMUP   = 10    
N_RUNS     = 100   
RANDOM_SEED = 42
N_PCA = 150
N_SVD = 150
N_NMF = 100
HOG_CELL = 8
HOG_BINS = 9
BATCH_SIZE = 128
NO_EPOCHS = 150
PATIENCE = 25
LABEL_SMOOTHING = 0.02
LR_MAX = 0.001
WEIGHT_DECAY = 1e-4

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# CLAHE
# ============================================================
def apply_clahe(gray_uint8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_uint8)

# ============================================================
# LOAD DATA
# ============================================================
def load_folder(folder_path, le=None, fit_le=False):
    image_paths = list(paths.list_images(folder_path))
    flat_data, labels, raw_images = [], [], []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue
        raw_images.append(img)   
        img     = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img     = apply_clahe(img)
        img_f32 = img.astype("float32") / 255.0
        flat_data.append(img_f32.flatten())
        labels.append(image_path.split(os.path.sep)[-2])
    if fit_le:
        le = LabelEncoder()
        le.fit(labels)
    y = le.transform(labels)
    return (np.array(flat_data, dtype="float32"),
            y, le, raw_images)

# ============================================================
# FEATURE EXTRACTORS — fitted on training data
# ============================================================
def fit_extractors(X_train):
    """Fit all extractors on training data once."""
    sc_pca = StandardScaler()
    pca    = PCA(n_components=N_PCA, whiten=True,
                  random_state=RANDOM_SEED)
    pca.fit(sc_pca.fit_transform(X_train))

    sc_svd = StandardScaler()
    svd    = TruncatedSVD(n_components=N_SVD,
                           random_state=RANDOM_SEED)
    svd.fit(sc_svd.fit_transform(X_train))

    nmf    = NMF(n_components=N_NMF, init="nndsvda",
                  max_iter=300, random_state=RANDOM_SEED)
    nmf_feats = nmf.fit_transform(X_train)
    sc_nmf = StandardScaler()
    sc_nmf.fit(nmf_feats)

    def hog_features(X):
        feats = []
        for flat in X:
            img = (flat.reshape(IMAGE_SIZE,
                                 IMAGE_SIZE)*255).astype(np.uint8)
            cs  = (HOG_CELL, HOG_CELL)
            bs  = (HOG_CELL*2, HOG_CELL*2)
            bst = (HOG_CELL, HOG_CELL)
            h   = cv2.HOGDescriptor(
                      (IMAGE_SIZE,IMAGE_SIZE),
                      bs, bst, cs, HOG_BINS)
            feats.append(h.compute(img).flatten())
        return np.array(feats, dtype="float32")
    sc_hog = StandardScaler()
    sc_hog.fit(hog_features(X_train))

    return (pca, sc_pca, svd, sc_svd,
            nmf, sc_nmf, sc_hog, hog_features)

def extract_features_single(flat_img, extractors):
    """Extract features for ONE image."""
    pca, sc_pca, svd, sc_svd, nmf, sc_nmf, sc_hog, hog_fn = \
        extractors
    x = flat_img.reshape(1, -1)

    p = pca.transform(sc_pca.transform(x))
    s = svd.transform(sc_svd.transform(x))
    n = sc_nmf.transform(nmf.transform(x))
    h = sc_hog.transform(hog_fn(x))

    def l2n(X):
        n_ = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(n_ == 0, 1, n_)

    return np.concatenate([l2n(p), l2n(s),
                            l2n(n), l2n(h)], axis=1)

# ============================================================
# MLP MODEL
# ============================================================
def build_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, use_bias=False),
        BatchNormalization(),
        tf.keras.layers.Activation("swish"),
        Dropout(0.25),
        Dense(256, use_bias=False),
        BatchNormalization(),
        tf.keras.layers.Activation("swish"),
        Dropout(0.25),
        Dense(num_classes, activation="softmax",
               dtype="float32")
    ])
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
                 label_smoothing=LABEL_SMOOTHING),
        optimizer=AdamW(learning_rate=LR_MAX,
                         weight_decay=WEIGHT_DECAY),
        metrics=["accuracy"])
    return model

def cosine_lr(epoch):
    t = epoch % 40
    return float(1e-6 + 0.5*(LR_MAX-1e-6)*
                 (1+math.cos(math.pi*t/40)))

# ============================================================
# MEASURE PREPROCESSING TIME
# ============================================================
def measure_preprocessing(raw_images):
    """Measure CLAHE + resize + normalize per image."""
    times = []
    # Warm-up
    for img in raw_images[:N_WARMUP]:
        _ = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    # Measure
    for img in raw_images[:N_RUNS]:
        t0  = time.perf_counter()
        img2 = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = apply_clahe(img2)
        _    = img2.astype("float32") / 255.0
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

# ============================================================
# MEASURE FEATURE EXTRACTION TIME
# ============================================================
def measure_feature_extraction(X_flat, extractors):
    """Measure PCA+SVD+NMF+HOG per image."""
    times = []
    # Warm-up
    for i in range(N_WARMUP):
        _ = extract_features_single(X_flat[i], extractors)
    # Measure
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        _  = extract_features_single(X_flat[i], extractors)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

# ============================================================
# MEASURE CLASSIFICATION TIME
# ============================================================
def measure_classification(model, X_feat):
    """Measure MLP forward pass per image."""
    times = []
    # Warm-up
    for i in range(N_WARMUP):
        _ = model.predict(X_feat[i:i+1], verbose=0)
    # Measure
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        _  = model.predict(X_feat[i:i+1], verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

# ============================================================
# COUNT MODEL PARAMETERS
# ============================================================
def count_params(model):
    total     = model.count_params()
    trainable = sum([tf.size(w).numpy()
                     for w in model.trainable_weights])
    return total, trainable

# ============================================================
# MAIN
# ============================================================
print("\n" + "="*60)
print("  INFERENCE TIME MEASUREMENT")
print("="*60)

# Load data
X_dev, y_dev, le, raw_dev = load_folder(
    "Images/Training", fit_le=True)
X_test, y_test, _, raw_test = load_folder(
    "Images/Testing", le=le)
num_classes = len(le.classes_)

print(f"\n  Fitting feature extractors on training data...")
extractors = fit_extractors(X_dev)

print(f"  Building and training MLP...")
X_dev_feat  = np.vstack([
    extract_features_single(X_dev[i], extractors)
    for i in range(len(X_dev))])
X_test_feat = np.vstack([
    extract_features_single(X_test[i], extractors)
    for i in range(len(X_test))])

cw_array = compute_class_weight("balanced",
                                  classes=np.unique(y_dev),
                                  y=y_dev)
cw_dict  = dict(enumerate(cw_array))

model = build_model(X_dev_feat.shape[1], num_classes)
y_dev_oh = tf.keras.utils.to_categorical(y_dev, num_classes)
lr_cb = tf.keras.callbacks.LearningRateScheduler(
    cosine_lr, verbose=0)
es_cb = EarlyStopping(monitor="val_accuracy",
                       patience=PATIENCE,
                       restore_best_weights=True,
                       verbose=0)
model.fit(X_dev_feat, y_dev_oh,
          batch_size=BATCH_SIZE, epochs=NO_EPOCHS,
          validation_split=0.10,
          class_weight=cw_dict,
          callbacks=[lr_cb, es_cb], verbose=0)

total_params, train_params = count_params(model)

# ── Measure latency ──────────────────────────────────────────
print(f"\n  Measuring preprocessing time "
      f"({N_WARMUP} warmup, {N_RUNS} runs)...")
pre_mean, pre_std = measure_preprocessing(raw_test)

print(f"  Measuring feature extraction time...")
feat_mean, feat_std = measure_feature_extraction(
    X_test, extractors)

print(f"  Measuring classification time...")
cls_mean, cls_std = measure_classification(
    model, X_test_feat)

total_mean = pre_mean + feat_mean + cls_mean
total_std  = np.sqrt(pre_std**2 + feat_std**2 + cls_std**2)

# ── Results ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  INFERENCE LATENCY RESULTS (per image, ms)")
print(f"{'='*60}")
print(f"  Preprocessing    : "
      f"{pre_mean:.3f} ± {pre_std:.3f} ms")
print(f"  Feature extract  : "
      f"{feat_mean:.3f} ± {feat_std:.3f} ms")
print(f"  Classification   : "
      f"{cls_mean:.3f} ± {cls_std:.3f} ms")
print(f"  {'─'*40}")
print(f"  Total latency    : "
      f"{total_mean:.3f} ± {total_std:.3f} ms")
print(f"\n  Model parameters : {total_params:,} total")
print(f"                     {train_params:,} trainable")
print(f"{'='*60}")

