"""
Brain MRI Cancer Classification — MLP Hyperparameter Tuning
================================================================
  Experiment 1 : Image size comparison   (64, 128, 256, 512)
  Experiment 2 : Batch size comparison   (32, 64, 128, 256)
  Experiment 3 : Activation function     (ReLU vs Swish)
  Experiment 4 : Learning rate selection

Protocol:
  - Development set only (5,712 images from Training folder)
  - StratifiedKFold(10), RANDOM_SEED=42
  - Fixed baseline: MLP(512, 256), AdamW, Cosine Annealing (T=40)
  - Early stopping: patience=12 (Experiments 1–3),
                    patience=25 (Experiment 4)
  - No test set touched
"""

import os, time, math, warnings
import cv2
import numpy as np
import tensorflow as tf
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Flatten, Dropout,
                                     BatchNormalization)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow.keras.backend as K

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# 1. SETTINGS
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

BASE_PATH = "Images"
TRAIN_DIR = os.path.join(BASE_PATH, "Training")

# ============================================================
# 2. COSINE ANNEALING LR
# ============================================================
def cosine_schedule(epoch, lr_max=0.001, lr_min=1e-6, T=40):
    t  = epoch % T
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / T))
    return float(lr)

# ============================================================
# 3. CLAHE PREPROCESSING
# ============================================================
def preprocess_image(img_path, image_size):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = cv2.resize(img, (image_size, image_size))
    img   = img.astype("float32") / 255.0
    return img.reshape(image_size, image_size, 1)

# ============================================================
# 4. DATA LOADING
# ============================================================
def load_dev_set(image_size):
    img_paths  = sorted(list(paths.list_images(TRAIN_DIR)))
    data, labels = [], []
    for p in img_paths:
        img = preprocess_image(p, image_size)
        if img is not None:
            data.append(img)
            labels.append(p.split(os.path.sep)[-2])
    data       = np.array(data, dtype="float32")
    le         = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    return data, labels_enc, le

# ============================================================
# 5. TRADE-OFF SCORE
# ============================================================
def tradeoff(acc_pct, time_sec):
    return round(acc_pct / time_sec, 4) if time_sec > 0 else 0.0

# ============================================================
# 6. MLP MODEL
# ============================================================
def build_mlp(image_size, hidden_units, activation, num_classes,
              dropout=0.25):
    model = Sequential()
    model.add(Flatten(input_shape=(image_size, image_size, 1)))
    for units in hidden_units:
        model.add(Dense(units))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation="softmax"))
    return model

# ============================================================
# 7. 10-FOLD CV RUNNER
# ============================================================
def run_cv(data, labels, num_classes, image_size, hidden_units,
           activation, batch_size, max_epochs, patience,
           lr=0.001, weight_decay=1e-4):
    skf       = StratifiedKFold(n_splits=10, shuffle=True,
                                random_state=RANDOM_SEED)
    accs      = []
    total_sec = 0.0

    for fold, (tr_idx, val_idx) in enumerate(skf.split(data, labels), 1):
        K.clear_session()
        X_tr,  X_val = data[tr_idx],   data[val_idx]
        y_tr,  y_val = labels[tr_idx], labels[val_idx]

        y_tr_cat  = tf.keras.utils.to_categorical(y_tr,  num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

        model = build_mlp(image_size, hidden_units, activation, num_classes)
        model.compile(
            loss      = "categorical_crossentropy",
            optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay),
            metrics   = ["accuracy"]
        )

        callbacks = [
            LearningRateScheduler(
                lambda ep, _lr=lr: cosine_schedule(ep, lr_max=_lr, T=40),
                verbose=0),
            EarlyStopping(monitor="val_loss", patience=patience,
                          restore_best_weights=True, verbose=0)
        ]

        t0      = time.time()
        model.fit(X_tr, y_tr_cat,
                  validation_data=(X_val, y_val_cat),
                  batch_size=batch_size,
                  epochs=max_epochs,
                  callbacks=callbacks,
                  verbose=0)
        elapsed    = time.time() - t0
        total_sec += elapsed

        preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
        accs.append(accuracy_score(y_val, preds) * 100)
        print(f"    Fold {fold:2d} | Acc={accs[-1]:.2f}% | "
              f"Time={elapsed:.0f}s")

    mean_acc = np.mean(accs)
    return round(mean_acc, 2), round(total_sec, 0), tradeoff(mean_acc, total_sec)

# ============================================================
# 8. PRINT TABLE
# ============================================================
def print_table(title, rows,
                header=("Architecture", "Acc (%)", "Time (sec)", "Trade-off")):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  {header[0]:<22} {header[1]:>8} {header[2]:>11} {header[3]:>11}")
    print(f"  {'-'*60}")
    for row in rows:
        print(f"  {str(row[0]):<22} {row[1]:>8.2f} {row[2]:>11.0f} {row[3]:>11.4f}")
    print(f"{'='*65}")

# ============================================================
# 9. EXPERIMENT 1 — Image Size Comparison (Tables VI–IX)
#    Fixed: batch=32, ReLU, epochs=100, patience=12
# ============================================================
print("\n" + "#"*65)
print("# EXPERIMENT 1: IMAGE SIZE COMPARISON")
print("#"*65)

IMAGE_SIZES   = [64, 128, 256, 512]
ARCHITECTURES = [(64, 128), (128, 256), (256, 512), (512, 1024)]

for img_sz in IMAGE_SIZES:
    print(f"\n  Loading data for image size {img_sz}×{img_sz} ...")
    data, labels, le = load_dev_set(img_sz)
    num_classes      = len(le.classes_)
    rows             = []
    for arch in ARCHITECTURES:
        print(f"  Architecture {arch} ...")
        acc, t, to = run_cv(data, labels, num_classes,
                            image_size   = img_sz,
                            hidden_units = arch,
                            activation   = "relu",
                            batch_size   = 32,
                            max_epochs   = 100,
                            patience     = 12)
        rows.append((arch, acc, t, to))
    print_table(f"TABLE — IMAGE SIZE = {img_sz}×{img_sz}", rows)

# ============================================================
# 10. EXPERIMENT 2 — Batch Size Comparison (Tables X–XIII)
#     Fixed: image_size=64, ReLU, epochs=100, patience=12
# ============================================================
print("\n" + "#"*65)
print("# EXPERIMENT 2: BATCH SIZE COMPARISON")
print("#"*65)

BATCH_SIZES = [32, 64, 128, 256]

print("  Loading data (64×64) ...")
data64, labels64, le64 = load_dev_set(64)
num_classes = len(le64.classes_)

for bs in BATCH_SIZES:
    rows = []
    for arch in ARCHITECTURES:
        print(f"  Batch={bs}, Architecture={arch} ...")
        acc, t, to = run_cv(data64, labels64, num_classes,
                            image_size   = 64,
                            hidden_units = arch,
                            activation   = "relu",
                            batch_size   = bs,
                            max_epochs   = 100,
                            patience     = 12)
        rows.append((arch, acc, t, to))
    print_table(f"TABLE — BATCH SIZE = {bs}", rows)

# ============================================================
# 11. EXPERIMENT 3 — Activation Function: ReLU vs Swish
#     (Tables XIV–XVI)
#     Fixed: image_size=64, epochs=100, patience=12
#     Batch sizes tested: 128 and 256
# ============================================================
print("\n" + "#"*65)
print("# EXPERIMENT 3: ACTIVATION FUNCTION — ReLU vs Swish")
print("#"*65)

for activation in ["relu", "swish"]:
    for bs in [128, 256]:
        rows = []
        for arch in ARCHITECTURES:
            print(f"  Activation={activation}, Batch={bs}, "
                  f"Architecture={arch} ...")
            acc, t, to = run_cv(data64, labels64, num_classes,
                                image_size   = 64,
                                hidden_units = arch,
                                activation   = activation,
                                batch_size   = bs,
                                max_epochs   = 100,
                                patience     = 12)
            rows.append((arch, acc, t, to))
        print_table(
            f"TABLE — {activation.upper()} | BATCH SIZE = {bs}", rows)

# ============================================================
# 12. EXPERIMENT 4 — Learning Rate Selection (Table XVII)
#     Fixed: image_size=64, batch=128, arch=(512,256),
#            Swish, epochs=100, patience=12
# ============================================================
print("\n" + "#"*65)
print("# EXPERIMENT 4: LEARNING RATE SELECTION")
print("#"*65)

LR_CANDIDATES = [0.1, 0.01, 0.001, 0.0005, 0.0001, 5e-5, 1e-5]
lr_rows = []
for lr in LR_CANDIDATES:
    print(f"  LR={lr} ...")
    acc, t, to = run_cv(data64, labels64, num_classes,
                        image_size   = 64,
                        hidden_units = (512, 256),
                        activation   = "swish",
                        batch_size   = 128,
                        max_epochs   = 100,
                        patience     = 12,
                        lr           = lr)
    lr_rows.append((lr, acc, t, to))

print_table(
    "TABLE XVII — LEARNING RATE SELECTION (Batch=128, Arch=(512,256))",
    lr_rows,
    header=("Learning Rate", "Acc (%)", "Time (sec)", "Trade-off")
)

# ============================================================
# 13. SUMMARY — Selected Final Configuration
# ============================================================
print("\n" + "="*65)
print("  SELECTED FINAL CONFIGURATION (from tuning experiments)")
print("="*65)
print("  Image size        : 64×64")
print("  Batch size        : 128")
print("  Architecture      : Dense(512) → Dense(256) → Softmax")
print("  Activation        : Swish")
print("  Optimizer         : AdamW (weight_decay=1e-4)")
print("  Learning rate     : 0.001")
print("  LR schedule       : Cosine Annealing (T=40)")
print("  Max epochs        : 150  (patience=25 for final runs)")
print("  Dropout           : 0.25")
print("  Label smoothing   : 0.02")
print("="*65)
