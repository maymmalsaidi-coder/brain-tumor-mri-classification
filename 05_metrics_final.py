"""
Brain MRI
=========================================

Methods compared in CV:
  1.  MLP baseline
  2.  PCA + MLP
  3.  SVD + MLP
  4.  NMF + MLP
  5.  Tucker + MLP
  6.  HOG + MLP
  7.  PCA + SVD + MLP
  8.  PCA + NMF + MLP
  9.  SVD + HOG + MLP
  10. PCA + SVD + NMF + MLP
  11. PCA + SVD + NMF + HOG + MLP

Dataset : Images/Training/<class>/ → development set (5,712)
          Images/Testing/<class>/  → final test set  (1,311)
"""

import os
import cv2
import numpy as np
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
from sklearn.metrics import (accuracy_score, precision_score,
                               recall_score, f1_score,
                               roc_auc_score, confusion_matrix)
from imutils import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import math, time, warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. SETTINGS
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
TUCKER_R1       = 20
TUCKER_R2       = 20
HOG_CELL        = 8
HOG_BINS        = 9

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 2. CLAHE
# ============================================================
def apply_clahe(gray_uint8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_uint8)

# ============================================================
# 3. DATA LOADING — STRICTLY SEPARATED
# ============================================================
def load_folder(folder_path, le=None, fit_le=False):
    """Load images from one folder. Returns flat + 2D arrays."""
    image_paths = list(paths.list_images(folder_path))
    flat_data, labels = [], []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue
        img     = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img     = apply_clahe(img)
        img_f32 = img.astype("float32") / 255.0
        flat_data.append(img_f32.flatten())
        labels.append(image_path.split(os.path.sep)[-2])

    if fit_le:
        le = LabelEncoder()
        le.fit(labels)
    y      = le.transform(labels)
    X_flat = np.array(flat_data, dtype="float32")
    return X_flat, y, le


def load_dataset(base_path):
    """
    Load development and test sets SEPARATELY.
    Test set is stored but NOT used until Phase 2.
    """
    dev_dir  = os.path.join(base_path, "Training")   
    test_dir = os.path.join(base_path, "Testing")    

    # Development set — fit LabelEncoder here
    X_dev, y_dev, le = load_folder(dev_dir, fit_le=True)
    # Test set — use same LabelEncoder, never fit on test
    X_test, y_test, _ = load_folder(test_dir, le=le)

    print("\n" + "="*65)
    print("  DATASET — Strictly Separated Protocol")
    print("="*65)
    print(f"  Development set : {len(y_dev):>5} images "
          f"← K-Fold CV applied here")
    print(f"  Final test set  : {len(y_test):>5} images "
          f"← UNTOUCHED until Phase 2")
    print(f"  Classes         : {list(le.classes_)}")
    print("="*65 + "\n")
    print("  Class distribution:")
    for i, c in enumerate(le.classes_):
        print(f"  {c:<30} "
              f"dev={np.sum(y_dev==i):>4}  "
              f"test={np.sum(y_test==i):>4}")
    print()
    return X_dev, y_dev, X_test, y_test, le

# ============================================================
# 4. FEATURE EXTRACTORS  (fit on train split only)
# ============================================================
def feat_pca(Xtr, Xte):
    sc = StandardScaler()
    m  = PCA(n_components=N_PCA, whiten=True,
              random_state=RANDOM_SEED)
    return (m.fit_transform(sc.fit_transform(Xtr)),
            m.transform(sc.transform(Xte)),
            (m, sc))

def feat_svd(Xtr, Xte):
    sc = StandardScaler()
    m  = TruncatedSVD(n_components=N_SVD,
                       random_state=RANDOM_SEED)
    return (m.fit_transform(sc.fit_transform(Xtr)),
            m.transform(sc.transform(Xte)),
            (m, sc))

def feat_nmf(Xtr, Xte):
    m  = NMF(n_components=N_NMF, init="nndsvda",
              max_iter=300, random_state=RANDOM_SEED)
    sc = StandardScaler()
    Xtr_out = sc.fit_transform(m.fit_transform(Xtr))
    Xte_out = sc.transform(m.transform(Xte))
    return Xtr_out, Xte_out, (m, sc)

def feat_tucker(Xtr, Xte):
    imgs_tr = Xtr.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    imgs_te = Xte.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    _, H, W   = imgs_tr.shape
    mode1     = imgs_tr.transpose(1, 0, 2).reshape(H, -1)
    U_h, _, _ = np.linalg.svd(mode1, full_matrices=False)
    U_h       = U_h[:, :TUCKER_R1]
    mode2     = imgs_tr.transpose(2, 0, 1).reshape(W, -1)
    U_w, _, _ = np.linalg.svd(mode2, full_matrices=False)
    U_w       = U_w[:, :TUCKER_R2]
    proj      = lambda X: np.array(
        [(U_h.T @ img @ U_w).flatten() for img in X],
        dtype="float32")
    sc        = StandardScaler()
    Xtr_out   = sc.fit_transform(proj(imgs_tr))
    Xte_out   = sc.transform(proj(imgs_te))
    return Xtr_out, Xte_out, (U_h, U_w, sc)

def feat_hog(Xtr, Xte):
    def extract(img_flat):
        img_2d = img_flat.reshape(IMAGE_SIZE, IMAGE_SIZE)
        img_u8 = (img_2d * 255).astype(np.uint8)
        cs  = (HOG_CELL, HOG_CELL)
        bs  = (HOG_CELL*2, HOG_CELL*2)
        bst = (HOG_CELL, HOG_CELL)
        hog = cv2.HOGDescriptor(
                  (IMAGE_SIZE, IMAGE_SIZE), bs, bst, cs, HOG_BINS)
        return hog.compute(img_u8).flatten().astype("float32")
    sc  = StandardScaler()
    Htr = np.array([extract(i) for i in Xtr], dtype="float32")
    Hte = np.array([extract(i) for i in Xte], dtype="float32")
    return sc.fit_transform(Htr), sc.transform(Hte), sc

def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)

def get_features(method, Xtr, Xte):
    """Build features. Fit on Xtr only."""
    if method == "none":
        sc = StandardScaler()
        return sc.fit_transform(Xtr), sc.transform(Xte)

    parts_tr, parts_te = [], []

    if "pca" in method:
        a, b, _ = feat_pca(Xtr, Xte)
        parts_tr.append(l2n(a)); parts_te.append(l2n(b))
    if "svd" in method:
        a, b, _ = feat_svd(Xtr, Xte)
        parts_tr.append(l2n(a)); parts_te.append(l2n(b))
    if "nmf" in method:
        a, b, _ = feat_nmf(Xtr, Xte)
        parts_tr.append(l2n(a)); parts_te.append(l2n(b))
    if "tucker" in method:
        a, b, _ = feat_tucker(Xtr, Xte)
        parts_tr.append(l2n(a)); parts_te.append(l2n(b))
    if "hog" in method:
        a, b, _ = feat_hog(Xtr, Xte)
        parts_tr.append(l2n(a)); parts_te.append(l2n(b))

    return (np.concatenate(parts_tr, axis=1),
            np.concatenate(parts_te, axis=1))

# ============================================================
# 5. MLP MODEL  
# ============================================================
def build_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        #Dense(1024, use_bias=False),
        #BatchNormalization(),
        #tf.keras.layers.Activation("swish"),
        #Dropout(0.25),
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
    return float(lr_min + 0.5*(lr_max - lr_min) *
                 (1 + math.cos(math.pi * t / T)))

# ============================================================
# 6. METRICS
# ============================================================
def compute_metrics(y_true, y_pred_labels,
                    y_pred_proba, num_classes):
    acc       = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels,
                                 average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred_labels,
                              average="macro", zero_division=0)
    f1        = f1_score(y_true, y_pred_labels,
                          average="macro", zero_division=0)
    try:
        y_oh  = tf.keras.utils.to_categorical(y_true, num_classes)
        auc   = roc_auc_score(y_oh, y_pred_proba,
                               multi_class="ovr", average="macro")
    except Exception:
        auc   = float("nan")
    cm = confusion_matrix(y_true, y_pred_labels)
    far_list, frr_list = [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - tp - fn - fp
        frr_list.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
        far_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return {"acc": acc, "precision": precision,
            "recall": recall, "f1": f1, "auc": auc,
            "far": float(np.mean(far_list)),
            "frr": float(np.mean(frr_list)),
            "cm": cm}

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================
def plot_cm(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=cm, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues", linewidths=0.5,
                linecolor="white", vmin=0, vmax=1,
                cbar_kws={"label": "Normalised rate"}, ax=ax)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0,  labelsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.abspath(save_path)}")

# ============================================================
# 8. K-Fold CV on Development Set
# ============================================================
def phase1_cv(method, X_dev, y_dev, num_classes, cw_dict):
    
    skfold     = StratifiedKFold(n_splits=NUM_FOLDS,
                                  shuffle=True,
                                  random_state=RANDOM_SEED)
    fold_accs  = []
    t_start    = time.time()

    for fold_no, (tr_idx, val_idx) in enumerate(
            skfold.split(X_dev, y_dev), start=1):

        y_tr  = y_dev[tr_idx]
        y_val = y_dev[val_idx]

        # Features fit on training split only
        X_tr, X_val = get_features(method,
                                    X_dev[tr_idx],
                                    X_dev[val_idx])

        y_tr_oh  = tf.keras.utils.to_categorical(y_tr,  num_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)

        model = build_model(X_tr.shape[1], num_classes)
        lr_cb = LearningRateScheduler(
            lambda ep: cosine_lr(ep, T=40), verbose=0)
        es_cb = EarlyStopping(monitor="val_accuracy",
                               patience=PATIENCE,
                               restore_best_weights=True,
                               verbose=0)

        model.fit(X_tr, y_tr_oh,
                  batch_size=BATCH_SIZE, epochs=NO_EPOCHS,
                  validation_data=(X_val, y_val_oh),
                  class_weight=cw_dict,
                  callbacks=[lr_cb, es_cb], verbose=0)

        _, acc = model.evaluate(X_val, y_val_oh, verbose=0)
        fold_accs.append(acc * 100)

        print(f"    Fold {fold_no:>2} : {acc*100:.2f}%")

    total = time.time() - t_start
    mean  = float(np.mean(fold_accs))
    std   = float(np.std(fold_accs))
    print(f"    CV Mean : {mean:.2f}% ± {std:.2f}%  "
          f"({int(total//60)}m{total%60:.0f}s)")
    return mean, std, total

# ============================================================
# 9. Final Evaluation on Test Set
# ============================================================
def phase2_final(best_method, X_dev, y_dev,
                 X_test, y_test, num_classes,
                 cw_dict, class_names):
    """
    1. Retrain on ALL development images (5,712)
    2. Evaluate ONCE on test set (1,311)
    3. Report all academic metrics
    """
    print(f"\n{'='*65}")
    print(f"  PHASE 2 — Final Evaluation")
    print(f"  Best method : {best_method}")
    print(f"  Retraining on all {len(y_dev)} development images...")
    print(f"{'='*65}\n")

    t_start = time.time()

    # Build features on full development set
    # fit on dev → transform test (no leakage)
    X_dev_feat, X_test_feat = get_features(
        best_method, X_dev, X_test)

    print(f"  Feature dim : {X_dev_feat.shape[1]}")
    print(f"  Dev   : {len(X_dev_feat)}  "
          f"Test : {len(X_test_feat)}\n")

    y_dev_oh  = tf.keras.utils.to_categorical(y_dev,  num_classes)
    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)

    # Retrain on full development set
    model = build_model(X_dev_feat.shape[1], num_classes)
    lr_cb = LearningRateScheduler(
        lambda ep: cosine_lr(ep, T=40), verbose=0)
    es_cb = EarlyStopping(monitor="val_accuracy",
                           patience=PATIENCE,
                           restore_best_weights=True,
                           verbose=0)

    model.fit(X_dev_feat, y_dev_oh,
              batch_size=BATCH_SIZE, epochs=NO_EPOCHS,
              # Use 10% of dev as internal validation
              # for early stopping only
              validation_split=0.10,
              class_weight=cw_dict,
              callbacks=[lr_cb, es_cb], verbose=1)

    # FINAL EVALUATION — test set touched ONCE here
    print("\n  Evaluating on independent test set (1,311 images)...")
    y_pred_proba  = model.predict(X_test_feat, verbose=0)
    y_pred_labels = np.argmax(y_pred_proba, axis=1)

    metrics = compute_metrics(y_test, y_pred_labels,
                               y_pred_proba, num_classes)
    total   = time.time() - t_start

    # Trade-off = Acc% / Time
    tradeoff = (metrics["acc"] * 100) / total

    print(f"\n  {'─'*54}")
    print(f"  FINAL TEST SET RESULTS")
    print(f"  {'─'*54}")
    print(f"  Accuracy  : {metrics['acc']*100:.2f}%")
    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%")
    print(f"  F1        : {metrics['f1']*100:.2f}%")
    print(f"  AUC       : {metrics['auc']:.4f}")
    print(f"  FAR       : {metrics['far']*100:.4f}%")
    print(f"  FRR       : {metrics['frr']*100:.4f}%")
    print(f"  Trade-off : {tradeoff:.4f}")
    print(f"  Time      : {int(total//60)}m{total%60:.1f}s")
    print(f"  {'─'*54}")

    # Save confusion matrix
    cm_path = f"final_cm_{best_method}.png"
    plot_cm(metrics["cm"], class_names,
            f"Final Test Set — {best_method}\n"
            f"Accuracy = {metrics['acc']*100:.2f}%",
            cm_path)

    return metrics, tradeoff, total

# ============================================================
# 10. MAIN
# ============================================================
X_dev, y_dev, X_test, y_test, le = load_dataset("Images")
num_classes = len(le.classes_)
class_names = list(le.classes_)

cw_array  = compute_class_weight("balanced",
                                   classes=np.unique(y_dev),
                                   y=y_dev)
cw_dict   = dict(enumerate(cw_array))

METHODS = [
    "none",
    "pca", "svd", "nmf", "tucker", "hog",
    "pca_svd", "pca_nmf", "svd_hog",
    "pca_svd_nmf", "pca_svd_nmf_hog",
]

LABELS = {
    "none"           : "MLP baseline              ",
    "pca"            : "PCA + MLP                 ",
    "svd"            : "SVD + MLP                 ",
    "nmf"            : "NMF + MLP                 ",
    "tucker"         : "Tucker + MLP              ",
    "hog"            : "HOG + MLP                 ",
    "pca_svd"        : "PCA + SVD + MLP           ",
    "pca_nmf"        : "PCA + NMF + MLP           ",
    "svd_hog"        : "SVD + HOG + MLP           ",
    "pca_svd_nmf"    : "PCA + SVD + NMF + MLP     ",
    "pca_svd_nmf_hog": "PCA + SVD + NMF + HOG + MLP",
}

# ============================================================
# CV on development set (test set)
# ============================================================
print("\n" + "█"*65)
print("  PHASE 1 — Model Selection via 10-Fold CV")
print("  Development set ONLY (5,712 images)")
print("  Test set remains UNTOUCHED")
print("█"*65)

cv_results  = []
grand_start = time.time()

for method in METHODS:
    print(f"\n  ── {LABELS[method].strip()} ──")
    mean, std, t = phase1_cv(method, X_dev, y_dev,
                              num_classes, cw_dict)
    cv_results.append({"method": method,
                        "mean"  : mean,
                        "std"   : std,
                        "time"  : t})

# Select best method
cv_sorted   = sorted(cv_results,
                      key=lambda r: r["mean"], reverse=True)
best_method = cv_sorted[0]["method"]
best_cv_acc = cv_sorted[0]["mean"]

print("\n\n" + "="*65)
print("  PHASE 1 RESULTS — CV Accuracies (Development Set)")
print("="*65)
print(f"  {'Method':<30} {'CV Mean':>9} {'±Std':>7} {'Time':>10}")
print("-"*65)
for r in cv_sorted:
    marker = " ← SELECTED" if r["method"] == best_method else ""
    print(f"  {LABELS[r['method']]:<30}"
          f"  {r['mean']:>8.2f}%"
          f"  ±{r['std']:.2f}%"
          f"  {int(r['time']//60):>2}m{r['time']%60:04.1f}s"
          f"{marker}")
print("="*65)
print(f"\n  Selected best configuration: "
      f"{LABELS[best_method].strip()}")
print(f"  CV Accuracy: {best_cv_acc:.2f}%")
print(f"\n  Test set will now be used for the FIRST time...")

# ============================================================
# Final evaluation (test set)
# ============================================================
print("\n" + "█"*65)
print("  PHASE 2 — Final Evaluation on Independent Test Set")
print("  Test set touched for the FIRST and ONLY time")
print("█"*65)

final_metrics, tradeoff, final_time = phase2_final(
    best_method, X_dev, y_dev,
    X_test, y_test, num_classes,
    cw_dict, class_names)

total_elapsed = time.time() - grand_start


