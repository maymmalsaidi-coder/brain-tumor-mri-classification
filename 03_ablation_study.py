"""
Ablation Study 
==================================
Full model : PCA + SVD + NMF + HOG
Remove ONE component at a time:
  Without PCA : SVD + NMF + HOG
  Without SVD : PCA + NMF + HOG
  Without NMF : PCA + SVD + HOG
  Without HOG : PCA + SVD + NMF

All five configurations (full model + four ablations) are run
using identical folds, random seeds, classifier architecture,
and stopping criteria.
Statistical significance is assessed via paired t-test (α = 0.05).
"""

import os, cv2, numpy as np, math, time, warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout,
                                      BatchNormalization, Input)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (EarlyStopping,
                                         LearningRateScheduler)
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from scipy import stats
from imutils import paths
warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS — identical to main experiments
# ============================================================
IMAGE_SIZE      = 64
BATCH_SIZE      = 128
NO_EPOCHS       = 150
PATIENCE        = 25
LABEL_SMOOTHING = 0.02
LR_MAX          = 0.001
WEIGHT_DECAY    = 1e-4
NUM_FOLDS       = 10
RANDOM_SEED     = 42
N_PCA           = 150
N_SVD           = 150
N_NMF           = 100
HOG_CELL        = 8
HOG_BINS        = 9

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# CLAHE + LOAD
# ============================================================
def apply_clahe(gray_uint8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_uint8)

def load_dataset(base_path):
    train_dir   = os.path.join(base_path, "Training")
    test_dir    = os.path.join(base_path, "Testing")
    all_paths   = (list(paths.list_images(train_dir)) +
                   list(paths.list_images(test_dir)))
    flat_data, labels = [], []
    for p in all_paths:
        img = cv2.imread(p)
        if img is None: continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = apply_clahe(img)
        flat_data.append(img.astype("float32").flatten() / 255.0)
        labels.append(p.split(os.path.sep)[-2])
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"\n  Dataset loaded : {len(flat_data)} images  |  "
          f"Classes : {list(le.classes_)}\n")
    return np.array(flat_data, dtype="float32"), y, le

# ============================================================
# FEATURE EXTRACTORS
# ============================================================
def feat_pca(Xtr, Xte):
    sc = StandardScaler()
    m  = PCA(n_components=N_PCA, whiten=True,
              random_state=RANDOM_SEED)
    return (m.fit_transform(sc.fit_transform(Xtr)),
            m.transform(sc.transform(Xte)))

def feat_svd(Xtr, Xte):
    sc = StandardScaler()
    m  = TruncatedSVD(n_components=N_SVD,
                       random_state=RANDOM_SEED)
    return (m.fit_transform(sc.fit_transform(Xtr)),
            m.transform(sc.transform(Xte)))

def feat_nmf(Xtr, Xte):
    m  = NMF(n_components=N_NMF, init="nndsvda",
              max_iter=300, random_state=RANDOM_SEED)
    sc = StandardScaler()
    return (sc.fit_transform(m.fit_transform(Xtr)),
            sc.transform(m.transform(Xte)))

def feat_hog(Xtr, Xte):
    def extract(flat):
        img = (flat.reshape(IMAGE_SIZE, IMAGE_SIZE)*255).astype(np.uint8)
        cs  = (HOG_CELL, HOG_CELL)
        bs  = (HOG_CELL*2, HOG_CELL*2)
        bst = (HOG_CELL, HOG_CELL)
        h   = cv2.HOGDescriptor(
                  (IMAGE_SIZE,IMAGE_SIZE), bs, bst, cs, HOG_BINS)
        return h.compute(img).flatten().astype("float32")
    sc  = StandardScaler()
    Htr = np.array([extract(i) for i in Xtr], dtype="float32")
    Hte = np.array([extract(i) for i in Xte], dtype="float32")
    return sc.fit_transform(Htr), sc.transform(Hte)

def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)

def get_features(method, Xtr, Xte):
    """Build concatenated feature vector for a given ablation config."""
    parts_tr, parts_te = [], []
    if "pca" in method:
        a, b = feat_pca(Xtr, Xte)
        parts_tr.append(l2n(a)); parts_te.append(l2n(b))
    if "svd" in method:
        a, b = feat_svd(Xtr, Xte)
        parts_tr.append(l2n(a)); parts_te.append(l2n(b))
    if "nmf" in method:
        a, b = feat_nmf(Xtr, Xte)
        parts_tr.append(l2n(a)); parts_te.append(l2n(b))
    if "hog" in method:
        a, b = feat_hog(Xtr, Xte)
        parts_tr.append(l2n(a)); parts_te.append(l2n(b))
    return (np.concatenate(parts_tr, axis=1),
            np.concatenate(parts_te, axis=1))

# ============================================================
# MLP MODEL — identical to main experiments
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
        Dense(num_classes, activation="softmax", dtype="float32")
    ])
    model.compile(
        loss      = tf.keras.losses.CategoricalCrossentropy(
                        label_smoothing=LABEL_SMOOTHING),
        optimizer = AdamW(learning_rate=LR_MAX,
                           weight_decay=WEIGHT_DECAY),
        metrics   = ["accuracy"]
    )
    return model

def cosine_lr(epoch, lr_max=LR_MAX, lr_min=1e-6, T=40):
    t = epoch % T
    return float(lr_min + 0.5*(lr_max-lr_min)*
                 (1+math.cos(math.pi*t/T)))

# ============================================================
# RUN ONE ABLATION CONFIGURATION
# ============================================================
def run_ablation(method_name, label, X_flat, y,
                  num_classes, cw_dict):
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT : {label}")
    print(f"{'#'*60}")

    skfold = StratifiedKFold(n_splits=NUM_FOLDS,
                              shuffle=True,
                              random_state=RANDOM_SEED)
    fold_accs = []
    t_start   = time.time()

    for fold_no, (tr_idx, te_idx) in enumerate(
            skfold.split(X_flat, y), start=1):

        y_tr = y[tr_idx]; y_te = y[te_idx]
        X_tr, X_te = get_features(method_name,
                                   X_flat[tr_idx],
                                   X_flat[te_idx])

        if fold_no == 1:
            print(f"  Feature dim : {X_tr.shape[1]}\n")

        y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes)
        y_te_oh = tf.keras.utils.to_categorical(y_te, num_classes)

        model = build_model(X_tr.shape[1], num_classes)
        lr_cb = LearningRateScheduler(
            lambda ep: cosine_lr(ep, T=40), verbose=0)
        es_cb = EarlyStopping(monitor="val_accuracy",
                               patience=PATIENCE,
                               restore_best_weights=True,
                               verbose=0)

        model.fit(X_tr, y_tr_oh,
                  batch_size=BATCH_SIZE, epochs=NO_EPOCHS,
                  validation_data=(X_te, y_te_oh),
                  class_weight=cw_dict,
                  callbacks=[lr_cb, es_cb], verbose=0)

        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        acc    = accuracy_score(y_te, y_pred) * 100
        fold_accs.append(acc)
        print(f"  Fold {fold_no:>2} : {acc:.2f}%")

    total = time.time() - t_start
    mean  = float(np.mean(fold_accs))
    std   = float(np.std(fold_accs))
    print(f"\n  Mean : {mean:.2f}% ± {std:.2f}%  "
          f"({int(total//60)}m{total%60:.0f}s)")

    return {"label"    : label,
            "method"   : method_name,
            "accs"     : fold_accs,
            "mean"     : mean,
            "std"      : std,
            "time_sec" : total}

# ============================================================
# MAIN — run ALL five configurations
# ============================================================
X_flat, y, le = load_dataset("Images")
num_classes   = len(le.classes_)
cw_array      = compute_class_weight("balanced",
                                      classes=np.unique(y), y=y)
cw_dict       = dict(enumerate(cw_array))

EXPERIMENTS = [
    ("pca_svd_nmf_hog", "Full (PCA+SVD+NMF+HOG)"),
    ("svd_nmf_hog",     "Without PCA (SVD+NMF+HOG)"),
    ("pca_nmf_hog",     "Without SVD (PCA+NMF+HOG)"),
    ("pca_svd_hog",     "Without NMF (PCA+SVD+HOG)"),
    ("pca_svd_nmf",     "Without HOG (PCA+SVD+NMF)"),
]

all_results = {}
for method, label in EXPERIMENTS:
    r = run_ablation(method, label, X_flat, y,
                      num_classes, cw_dict)
    all_results[method] = r

# ============================================================
# STATISTICAL SIGNIFICANCE — paired t-test vs full model
# ============================================================
full_accs = np.array(all_results["pca_svd_nmf_hog"]["accs"])
full_mean = all_results["pca_svd_nmf_hog"]["mean"]

print("\n\n" + "="*70)
print("  ABLATION STUDY — Final Results Table")
print("="*70)
print(f"  {'Experiment':<32} {'Mean':>7} {'±Std':>6} "
      f"{'Δ vs Full':>10} {'p-value':>10} {'Sig?':>6}")
print("-"*70)

for method, label in EXPERIMENTS:
    r     = all_results[method]
    accs  = np.array(r["accs"])
    mean  = r["mean"]
    std   = r["std"]
    delta = mean - full_mean

    if method == "pca_svd_nmf_hog":
        print(f"  {label:<32}"
              f"  {mean:>6.2f}%"
              f"  ±{std:.2f}%"
              f"  {'—':>10}"
              f"  {'—':>10}"
              f"  {'—':>6}")
    else:
        t_stat, p_val = stats.ttest_rel(full_accs, accs)
        sig = "Yes" if p_val < 0.05 else "No"
        print(f"  {label:<32}"
              f"  {mean:>6.2f}%"
              f"  ±{std:.2f}%"
              f"  {delta:>+9.2f}%"
              f"  {p_val:>10.4f}"
              f"  {sig:>6}")

print("="*70)

