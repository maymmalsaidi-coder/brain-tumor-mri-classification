"""
Brain MRI
======================================================
1. FAR and FRR computed via binary collapse:
   Tumor    = Glioma + Meningioma + Pituitary (Positive)
   No Tumor = No Tumor (Negative)

   FP = no-tumor classified as tumor
   FN = tumor classified as no-tumor
   FAR = FP / (FP + TN)
   FRR = FN / (FN + TP)

2. Per-class metrics:
   - Sensitivity (Recall) per class
   - Specificity per class
   - Precision per class
   - F1 per class
   - One-vs-rest AUC per class
   - Macro and weighted averages

3. Two confusion matrices:
   - 4-class confusion matrix
   - Binary (Tumor vs No-Tumor) confusion matrix
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
from sklearn.metrics import (accuracy_score, precision_score,
                               recall_score, f1_score,
                               roc_auc_score, confusion_matrix)
from imutils import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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
HOG_CELL        = 8
HOG_BINS        = 9

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 2. CLAHE
# ============================================================
def apply_clahe(gray_uint8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_uint8)

# ============================================================
# 3. DATA LOADING — strictly separated
# ============================================================
def load_folder(folder_path, le=None, fit_le=False):
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
    dev_dir  = os.path.join(base_path, "Training")
    test_dir = os.path.join(base_path, "Testing")
    X_dev,  y_dev,  le = load_folder(dev_dir,  fit_le=True)
    X_test, y_test, _  = load_folder(test_dir, le=le)
    print(f"\n  Development : {len(y_dev)} images")
    print(f"  Test set    : {len(y_test)} images")
    print(f"  Classes     : {list(le.classes_)}")
    return X_dev, y_dev, X_test, y_test, le

# ============================================================
# 4. FIND NO-TUMOR CLASS AUTOMATICALLY
# ============================================================
def find_no_tumor_idx(class_names):
    """
    Find the no-tumor class index automatically
    by looking for class names containing 'no' or
    not containing typical tumor keywords.
    """
    tumor_keywords = ["glioma", "meningioma", "pituitary"]

    for i, c in enumerate(class_names):
        c_lower = c.lower().replace("_", "").replace(" ", "")
        # Check if it's NOT a tumor class
        is_tumor = any(kw in c_lower for kw in tumor_keywords)
        if not is_tumor:
            print(f"  No-tumor class found: '{c}' "
                  f"(index={i})")
            return i

    # Fallback: look for 'no' in name
    for i, c in enumerate(class_names):
        if "no" in c.lower():
            print(f"  No-tumor class found: '{c}' "
                  f"(index={i})")
            return i

    raise ValueError(
        f"Could not find no-tumor class in: {class_names}\n"
        f"Please set no_tumor_idx manually.")

# ============================================================
# 5. FEATURE EXTRACTORS
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
        img = (flat.reshape(IMAGE_SIZE,
                             IMAGE_SIZE)*255).astype(np.uint8)
        cs  = (HOG_CELL, HOG_CELL)
        bs  = (HOG_CELL*2, HOG_CELL*2)
        bst = (HOG_CELL, HOG_CELL)
        h   = cv2.HOGDescriptor(
                  (IMAGE_SIZE, IMAGE_SIZE),
                  bs, bst, cs, HOG_BINS)
        return h.compute(img).flatten().astype("float32")
    sc  = StandardScaler()
    Htr = np.array([extract(i) for i in Xtr],
                    dtype="float32")
    Hte = np.array([extract(i) for i in Xte],
                    dtype="float32")
    return sc.fit_transform(Htr), sc.transform(Hte)

def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)

def build_features(Xtr, Xte):
    p = feat_pca(Xtr, Xte)
    s = feat_svd(Xtr, Xte)
    n = feat_nmf(Xtr, Xte)
    h = feat_hog(Xtr, Xte)
    return (np.concatenate([l2n(p[0]), l2n(s[0]),
                             l2n(n[0]), l2n(h[0])], axis=1),
            np.concatenate([l2n(p[1]), l2n(s[1]),
                             l2n(n[1]), l2n(h[1])], axis=1))

# ============================================================
# 6. MLP MODEL
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
# 7. Binary Collapse
# ============================================================
def compute_binary_far_frr(y_true, y_pred, no_tumor_idx):
    y_bin_true = (y_true != no_tumor_idx).astype(int)
    y_bin_pred = (y_pred != no_tumor_idx).astype(int)

    cm = confusion_matrix(y_bin_true, y_bin_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    return FAR, FRR, TP, TN, FP, FN, y_bin_true, y_bin_pred

# ============================================================
# 8. PER-CLASS METRICS
# ============================================================
def compute_per_class_metrics(y_true, y_pred,
                               y_pred_proba,
                               class_names, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    results = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - tp - fn - fp

        sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        specificity = tn/(tn+fp) if (tn+fp) > 0 else 0.0
        precision   = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        f1          = (2*precision*sensitivity/
                       (precision+sensitivity)
                       if (precision+sensitivity) > 0
                       else 0.0)

        y_bin = (y_true == c).astype(int)
        try:
            auc = roc_auc_score(y_bin, y_pred_proba[:, c])
        except Exception:
            auc = float("nan")

        results[class_names[c]] = {
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision"  : precision,
            "F1"         : f1,
            "AUC_OvR"   : auc,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn
        }

    return results, cm

# ============================================================
# 9. CONFUSION MATRIX PLOTS
# ============================================================
def plot_4class_cm(cm, class_names, accuracy, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype(float) / cm.sum(axis=1,
                                         keepdims=True)
    sns.heatmap(cm_norm, annot=cm, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues", linewidths=0.5,
                linecolor="white", vmin=0, vmax=1,
                cbar_kws={"label": "Normalised rate"},
                ax=ax)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title(
        f"4-Class Confusion Matrix\n"
        f"PCA+SVD+NMF+HOG + MLP(256,512)\n"
        f"Accuracy = {accuracy*100:.2f}%",
        fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")

def plot_binary_cm(y_bin_true, y_bin_pred,
                    far, frr, save_path):
    cm   = confusion_matrix(y_bin_true, y_bin_pred)
    labs = ["No Tumor", "Tumor"]
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_norm = cm.astype(float) / cm.sum(axis=1,
                                         keepdims=True)
    sns.heatmap(cm_norm, annot=cm, fmt="d",
                xticklabels=labs, yticklabels=labs,
                cmap="Blues", linewidths=0.5,
                linecolor="white", vmin=0, vmax=1,
                cbar_kws={"label": "Normalised rate"},
                ax=ax)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title(
        f"Binary Confusion Matrix\n"
        f"Tumor vs No-Tumor\n"
        f"FAR = {far*100:.2f}%  |  "
        f"FRR = {frr*100:.2f}%",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")

# ============================================================
# 10. MAIN
# ============================================================
X_dev, y_dev, X_test, y_test, le = load_dataset("Images")
num_classes = len(le.classes_)
class_names = list(le.classes_)

# Auto-detect no-tumor class
no_tumor_idx = find_no_tumor_idx(class_names)

cw_array = compute_class_weight("balanced",
                                  classes=np.unique(y_dev),
                                  y=y_dev)
cw_dict  = dict(enumerate(cw_array))

# ── Phase 1: CV on development set ──────────────────────────
print("\n" + "="*55)
print("  PHASE 1 — 10-Fold CV on Development Set")
print("="*55)

skfold  = StratifiedKFold(n_splits=NUM_FOLDS,
                           shuffle=True,
                           random_state=RANDOM_SEED)
cv_accs = []

for fold_no, (tr_idx, val_idx) in enumerate(
        skfold.split(X_dev, y_dev), start=1):

    X_tr, X_val = build_features(X_dev[tr_idx],
                                   X_dev[val_idx])
    y_tr  = y_dev[tr_idx]
    y_val = y_dev[val_idx]

    y_tr_oh  = tf.keras.utils.to_categorical(y_tr,
                                               num_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val,
                                              num_classes)

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
    cv_accs.append(acc * 100)
    print(f"  Fold {fold_no:>2} : {acc*100:.2f}%")

cv_mean = np.mean(cv_accs)
cv_std  = np.std(cv_accs)
print(f"\n  CV Mean : {cv_mean:.2f}% ± {cv_std:.2f}%")

# ── Phase 2: Retrain on full dev + evaluate on test ─────────
print("\n" + "="*55)
print("  PHASE 2 — Final Evaluation on Test Set")
print("="*55)

X_dev_feat, X_test_feat = build_features(X_dev, X_test)
y_dev_oh = tf.keras.utils.to_categorical(y_dev,
                                           num_classes)

model = build_model(X_dev_feat.shape[1], num_classes)
lr_cb = LearningRateScheduler(
    lambda ep: cosine_lr(ep, T=40), verbose=0)
es_cb = EarlyStopping(monitor="val_accuracy",
                       patience=PATIENCE,
                       restore_best_weights=True,
                       verbose=0)

model.fit(X_dev_feat, y_dev_oh,
          batch_size=BATCH_SIZE, epochs=NO_EPOCHS,
          validation_split=0.10,
          class_weight=cw_dict,
          callbacks=[lr_cb, es_cb], verbose=1)

# Predictions
y_pred_proba  = model.predict(X_test_feat, verbose=0)
y_pred_labels = np.argmax(y_pred_proba, axis=1)

# ── Overall metrics ──────────────────────────────────────────
acc       = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels,
                             average="macro",
                             zero_division=0)
recall    = recall_score(y_test, y_pred_labels,
                          average="macro",
                          zero_division=0)
f1        = f1_score(y_test, y_pred_labels,
                      average="macro",
                      zero_division=0)
y_test_oh    = tf.keras.utils.to_categorical(y_test,
                                               num_classes)
auc_macro    = roc_auc_score(y_test_oh, y_pred_proba,
                               multi_class="ovr",
                               average="macro")
auc_weighted = roc_auc_score(y_test_oh, y_pred_proba,
                               multi_class="ovr",
                               average="weighted")

prec_w = precision_score(y_test, y_pred_labels,
                          average="weighted",
                          zero_division=0)
rec_w  = recall_score(y_test, y_pred_labels,
                       average="weighted",
                       zero_division=0)
f1_w   = f1_score(y_test, y_pred_labels,
                   average="weighted",
                   zero_division=0)

# ── CORRECT FAR/FRR via binary collapse ─────────────────────
(FAR, FRR, TP, TN, FP, FN,
 y_bin_true, y_bin_pred) = compute_binary_far_frr(
    y_test, y_pred_labels, no_tumor_idx)

# ── Per-class metrics ────────────────────────────────────────
per_class, cm_4class = compute_per_class_metrics(
    y_test, y_pred_labels, y_pred_proba,
    class_names, num_classes)

sens_macro = np.mean([m["Sensitivity"]
                       for m in per_class.values()])
spec_macro = np.mean([m["Specificity"]
                       for m in per_class.values()])

# ── Print all results ────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FINAL TEST SET RESULTS (n=1,311)")
print(f"{'='*65}")

print(f"\n  OVERALL METRICS:")
print(f"  Accuracy  (macro)    : {acc*100:.2f}%")
print(f"  Precision (macro)    : {precision*100:.2f}%")
print(f"  Recall    (macro)    : {recall*100:.2f}%")
print(f"  F1-Score  (macro)    : {f1*100:.2f}%")
print(f"  AUC       (macro)    : {auc_macro:.4f}")
print(f"  AUC       (weighted) : {auc_weighted:.4f}")
print(f"  Precision (weighted) : {prec_w*100:.2f}%")
print(f"  Recall    (weighted) : {rec_w*100:.2f}%")
print(f"  F1-Score  (weighted) : {f1_w*100:.2f}%")

tumor_classes = [c for c in class_names
                  if c != class_names[no_tumor_idx]]
no_tumor_name = class_names[no_tumor_idx]
print(f"\n  BINARY — Tumor vs No-Tumor:")
print(f"  Tumor    = {tumor_classes}")
print(f"  No-Tumor = {no_tumor_name}")
print(f"  TP = {TP}  TN = {TN}  FP = {FP}  FN = {FN}")
print(f"  FAR (False Positive Rate) : {FAR*100:.4f}%")
print(f"  FRR (False Negative Rate) : {FRR*100:.4f}%")

print(f"\n  PER-CLASS METRICS:")
print(f"  {'Class':<20} {'Sens%':>7} {'Spec%':>7} "
      f"{'Prec%':>7} {'F1%':>7} {'AUC_OvR':>9}")
print(f"  {'─'*60}")
for cls, m in per_class.items():
    print(f"  {cls:<20}"
          f"  {m['Sensitivity']*100:>6.2f}%"
          f"  {m['Specificity']*100:>6.2f}%"
          f"  {m['Precision']*100:>6.2f}%"
          f"  {m['F1']*100:>6.2f}%"
          f"  {m['AUC_OvR']:>8.4f}")
print(f"  {'─'*60}")
print(f"  {'Macro Average':<20}"
      f"  {sens_macro*100:>6.2f}%"
      f"  {spec_macro*100:>6.2f}%"
      f"  {precision*100:>6.2f}%"
      f"  {f1*100:>6.2f}%"
      f"  {auc_macro:>8.4f}")
print(f"  {'Weighted Average':<20}"
      f"  {rec_w*100:>6.2f}%"
      f"  {'—':>7}"
      f"  {prec_w*100:>6.2f}%"
      f"  {f1_w*100:>6.2f}%"
      f"  {auc_weighted:>8.4f}")

# ── Save confusion matrices ──────────────────────────────────
plot_4class_cm(cm_4class, class_names, acc,
               "confusion_matrix_4class.png")
plot_binary_cm(y_bin_true, y_bin_pred, FAR, FRR,
               "confusion_matrix_binary.png")

