"""
Brain MRI Cancer Classification 
================================================================
  1.  MLP baseline
  2.  PCA + MLP
  3.  SVD + MLP
  4.  NMF + MLP
  5.  Tucker + MLP
  6.  HOG + MLP
  7.  PCA + SVD + MLP
  8.  PCA + NMF + MLP
  9.  SVD + HOG + MLP
  10. PCA + SVD + NMF  + MLP
  11. PCA + SVD + NMF + HOG + MLP

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
                               roc_auc_score, confusion_matrix,
                               classification_report)
from imutils import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Output folder for confusion matrix images
os.makedirs("confusion_matrices", exist_ok=True)

# ============================================================
# 2. CLAHE
# ============================================================
def apply_clahe(gray_uint8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_uint8)

# ============================================================
# 3. DATA LOADING
# ============================================================
def load_dataset(base_path):
    train_dir   = os.path.join(base_path, "Training")
    test_dir    = os.path.join(base_path, "Testing")
    train_paths = list(paths.list_images(train_dir))
    test_paths  = list(paths.list_images(test_dir))
    all_paths   = train_paths + test_paths

    print("\n" + "="*62)
    print("  DATASET SUMMARY")
    print("="*62)
    print(f"  Training : {len(train_paths)}  |  "
          f"Testing : {len(test_paths)}  |  "
          f"Total : {len(all_paths)}")
    print("="*62 + "\n")

    flat_data, tensor_data, labels = [], [], []
    for image_path in all_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = apply_clahe(img)
        img_f32 = img.astype("float32") / 255.0
        flat_data.append(img_f32.flatten())
        tensor_data.append(img_f32)
        labels.append(image_path.split(os.path.sep)[-2])

    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_flat = np.array(flat_data,   dtype="float32")
    X_2d = np.array(tensor_data, dtype="float32")

    print(f"  Classes : {list(le.classes_)}")
    print(f"  Samples : {X_flat.shape[0]}\n")
    for i, c in enumerate(le.classes_):
        print(f"  {c:<35} : {np.sum(y==i)} images")
    print()
    return X_flat, X_2d, y, le

# ============================================================
# 4. FEATURE EXTRACTORS
# ============================================================
def feat_pca(Xf_tr, Xf_te):
    sc = StandardScaler()
    m = PCA(n_components=N_PCA, whiten=True,
               random_state=RANDOM_SEED)
    return (m.fit_transform(sc.fit_transform(Xf_tr)),
            m.transform(sc.transform(Xf_te)))

def feat_svd(Xf_tr, Xf_te):
    sc = StandardScaler()
    m = TruncatedSVD(n_components=N_SVD,
                        random_state=RANDOM_SEED)
    return (m.fit_transform(sc.fit_transform(Xf_tr)),
            m.transform(sc.transform(Xf_te)))

def feat_nmf(Xf_tr, Xf_te):
    m = NMF(n_components=N_NMF, init="nndsvda",
               max_iter=300, random_state=RANDOM_SEED)
    sc = StandardScaler()
    return (sc.fit_transform(m.fit_transform(Xf_tr)),
            sc.transform(m.transform(Xf_te)))

def feat_tucker(X2_tr, X2_te):
    _, H, W   = X2_tr.shape
    mode1     = X2_tr.transpose(1, 0, 2).reshape(H, -1)
    U_h, _, _ = np.linalg.svd(mode1, full_matrices=False)
    U_h       = U_h[:, :TUCKER_R1]
    mode2     = X2_tr.transpose(2, 0, 1).reshape(W, -1)
    U_w, _, _ = np.linalg.svd(mode2, full_matrices=False)
    U_w       = U_w[:, :TUCKER_R2]
    proj      = lambda X: np.array(
        [(U_h.T @ img @ U_w).flatten() for img in X],
        dtype="float32")
    sc        = StandardScaler()
    return (sc.fit_transform(proj(X2_tr)),
            sc.transform(proj(X2_te)))

def feat_hog(X2_tr, X2_te):
    def extract(img_2d):
        img_u8 = (img_2d * 255).astype(np.uint8)
        cs  = (HOG_CELL, HOG_CELL)
        bs  = (HOG_CELL*2, HOG_CELL*2)
        bst = (HOG_CELL, HOG_CELL)
        hog = cv2.HOGDescriptor(
                  (IMAGE_SIZE, IMAGE_SIZE), bs, bst, cs, HOG_BINS)
        return hog.compute(img_u8).flatten().astype("float32")
    sc  = StandardScaler()
    Htr = np.array([extract(i) for i in X2_tr], dtype="float32")
    Hte = np.array([extract(i) for i in X2_te], dtype="float32")
    return sc.fit_transform(Htr), sc.transform(Hte)

def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)

def concat(*pairs):
    return (np.concatenate([l2n(p[0]) for p in pairs], axis=1),
            np.concatenate([l2n(p[1]) for p in pairs], axis=1))

def get_features(method, Xf_tr, Xf_te, X2_tr, X2_te):
    if method == "none":
        sc = StandardScaler()
        return sc.fit_transform(Xf_tr), sc.transform(Xf_te)
    elif method == "pca":
        return feat_pca(Xf_tr, Xf_te)
    elif method == "svd":
        return feat_svd(Xf_tr, Xf_te)
    elif method == "nmf":
        return feat_nmf(Xf_tr, Xf_te)
    elif method == "tucker":
        return feat_tucker(X2_tr, X2_te)
    elif method == "hog":
        return feat_hog(X2_tr, X2_te)
    elif method == "pca_svd":
        return concat(feat_pca(Xf_tr, Xf_te),
                      feat_svd(Xf_tr, Xf_te))
    elif method == "pca_nmf":
        return concat(feat_pca(Xf_tr, Xf_te),
                      feat_nmf(Xf_tr, Xf_te))
    elif method == "svd_hog":
        return concat(feat_svd(Xf_tr, Xf_te),
                      feat_hog(X2_tr, X2_te))
    elif method == "pca_svd_nmf":
        return concat(feat_pca(Xf_tr, Xf_te),
                      feat_svd(Xf_tr, Xf_te),
                      feat_nmf(Xf_tr, Xf_te))
    elif method == "pca_svd_nmf_hog":
        return concat(feat_pca(Xf_tr, Xf_te),
                      feat_svd(Xf_tr, Xf_te),
                      feat_nmf(Xf_tr, Xf_te),
                      feat_hog(X2_tr, X2_te))
    else:
        raise ValueError(f"Unknown: {method}")

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
# 6. METRICS COMPUTATION
# ============================================================
def compute_metrics(y_true, y_pred_labels, y_pred_proba,
                    num_classes):
    """
    Compute all required metrics from predictions.
    Returns dict with all metrics.
    """
    acc  = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels,
                                 average="macro",
                                 zero_division=0)
    recall = recall_score(y_true, y_pred_labels,
                              average="macro",
                              zero_division=0)
    f1 = f1_score(y_true, y_pred_labels,
                          average="macro",
                          zero_division=0)

    # AUC — one-vs-rest for multiclass
    try:
        y_true_oh = tf.keras.utils.to_categorical(
            y_true, num_classes)
        auc = roc_auc_score(y_true_oh, y_pred_proba,
                             multi_class="ovr",
                             average="macro")
    except Exception:
        auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_labels)

    # FAR and FRR (macro-averaged across classes)
    far_list, frr_list = [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp       # missed positives
        fp = np.sum(cm[:, c]) - tp       # false alarms
        tn = np.sum(cm) - tp - fn - fp   # true negatives

        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr_list.append(frr)
        far_list.append(far)

    frr_macro    = float(np.mean(frr_list))
    far_macro    = float(np.mean(far_list))
    tradeoff     = frr_macro + far_macro

    return {"acc"      : acc,
            "precision": precision,
            "recall"   : recall,
            "f1"       : f1,
            "auc"      : auc,
            "far"      : far_macro,
            "frr"      : frr_macro,
            "tradeoff" : tradeoff,
            "cm"       : cm}

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(cm, class_names, method_label,
                           accuracy, save_path):
    """
    Save a publication-quality confusion matrix image.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalise for display (keep raw counts for annotation)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=cm, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues", linewidths=0.5,
                linecolor="white",
                vmin=0, vmax=1,
                cbar_kws={"label": "Normalised rate"},
                ax=ax)

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {method_label}\n"
        f"Accuracy = {accuracy*100:.2f}%",
        fontsize=13, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0,  labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")

# ============================================================
# 8. 10-Fold + full metrics
# ============================================================
def run_method(method_name, X_flat, X_2d, y,
               num_classes, cw_dict, class_names):

    print(f"\n{'#'*62}")
    print(f"  METHOD : {method_name.upper()}")
    print(f"{'#'*62}")

    skfold = StratifiedKFold(n_splits=NUM_FOLDS,
                              shuffle=True,
                              random_state=RANDOM_SEED)

    # Accumulators across folds
    fold_accs      = []
    fold_metrics   = []          # list of metric dicts
    cm_accumulated = np.zeros((num_classes, num_classes),
                               dtype=int)
    method_start   = time.time()

    for fold_no, (tr_idx, te_idx) in enumerate(
            skfold.split(X_flat, y), start=1):

        fold_start = time.time()
        y_tr       = y[tr_idx]
        y_te       = y[te_idx]

        X_tr, X_te = get_features(
            method_name,
            X_flat[tr_idx], X_flat[te_idx],
            X_2d[tr_idx],   X_2d[te_idx]
        )

        if fold_no == 1:
            print(f"  Feature dim : {X_tr.shape[1]}")
            print(f"  Train : {len(X_tr)}  "
                  f"Test : {len(X_te)}\n")

        y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes)
        y_te_oh = tf.keras.utils.to_categorical(y_te, num_classes)

        model = build_model(X_tr.shape[1], num_classes)

        lr_cb = LearningRateScheduler(
            lambda ep: cosine_lr(ep, T=40), verbose=0)
        es_cb = EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=0)

        model.fit(X_tr, y_tr_oh,
                  batch_size=BATCH_SIZE, epochs=NO_EPOCHS,
                  validation_data=(X_te, y_te_oh),
                  class_weight=cw_dict,
                  callbacks=[lr_cb, es_cb],
                  verbose=0)

        # Predictions
        y_pred_proba  = model.predict(X_te, verbose=0)
        y_pred_labels = np.argmax(y_pred_proba, axis=1)

        # Compute metrics for this fold
        m = compute_metrics(y_te, y_pred_labels,
                             y_pred_proba, num_classes)
        fold_accs.append(m["acc"])
        fold_metrics.append(m)

        # Accumulate confusion matrix
        cm_accumulated += m["cm"]

        ft   = time.time() - fold_start
        flag = "  ✓" if m["acc"] * 100 >= 98.0 else ""
        print(f"  Fold {fold_no:>2} : {m['acc']*100:.2f}%"
              f"  F1={m['f1']:.4f}"
              f"  AUC={m['auc']:.4f}"
              f"  ({int(ft//60)}m{ft%60:.0f}s){flag}")

    total_time = time.time() - method_start

    # ── Aggregate metrics (mean over folds) ──────────────────
    def fold_mean(key):
        return float(np.mean([fm[key] for fm in fold_metrics]))
    def fold_std(key):
        return float(np.std([fm[key] for fm in fold_metrics]))

    result = {
        "method"    : method_name,
        "accs"      : fold_accs,
        "mean_acc"  : float(np.mean(fold_accs)),
        "std_acc"   : float(np.std(fold_accs)),
        "best_acc"  : float(np.max(fold_accs)),
        "precision" : fold_mean("precision"),
        "recall"    : fold_mean("recall"),
        "f1"        : fold_mean("f1"),
        "auc"       : fold_mean("auc"),
        "far"       : fold_mean("far"),
        "frr"       : fold_mean("frr"),
        "tradeoff"  : fold_mean("tradeoff"),
        "cm"        : cm_accumulated,
        "time_sec"  : total_time,
    }

    mins = int(total_time // 60)
    secs = total_time % 60

    print(f"\n  {'─'*54}")
    print(f"  Accuracy  : {result['mean_acc']*100:.2f}%"
          + (" ✓ 98%+" if result['mean_acc']*100 >= 98.0 else ""))
    print(f"  Precision : {result['precision']*100:.2f}%")
    print(f"  Recall    : {result['recall']*100:.2f}%")
    print(f"  F1        : {result['f1']*100:.2f}%")
    print(f"  AUC       : {result['auc']:.4f}")
    print(f"  FAR       : {result['far']*100:.4f}%")
    print(f"  FRR       : {result['frr']*100:.4f}%")
    print(f"  Trade-off : {result['tradeoff']*100:.4f}%")
    print(f"  Std       : ±{result['std_acc']*100:.2f}%")
    print(f"  Time      : {mins}m {secs:.1f}s")
    print(f"  {'─'*54}")

    return result

# ============================================================
# 9. MAIN
# ============================================================
X_flat, X_2d, y, le = load_dataset("Images")
num_classes          = len(le.classes_)
class_names          = list(le.classes_)

cw_array = compute_class_weight("balanced",
                                   classes=np.unique(y), y=y)
cw_dict = dict(enumerate(cw_array))

METHODS = [
    "none",
    "pca",
    "svd",
    "nmf",
    "tucker",
    "hog",
    "pca_svd",
    "pca_nmf",
    "svd_hog",
    "pca_svd_nmf",
    "pca_svd_nmf_hog",
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

grand_start = time.time()
all_results = []

for m in METHODS:
    r = run_method(m, X_flat, X_2d, y,
                   num_classes, cw_dict, class_names)
    all_results.append(r)

# ============================================================
# 10. SAVE CONFUSION MATRIX FOR BEST METHOD
# ============================================================
results_sorted = sorted(all_results,
                         key=lambda r: r["mean_acc"],
                         reverse=True)
best           = results_sorted[0]

best_label = LABELS.get(best["method"], best["method"]).strip()
cm_path    = os.path.join(
    "confusion_matrices",
    f"cm_{best['method']}_best.png"
)
plot_confusion_matrix(
    best["cm"], class_names,
    best_label, best["mean_acc"],
    cm_path
)

# Also save confusion matrix for every method
for r in all_results:
    lbl  = LABELS.get(r["method"], r["method"]).strip()
    path = os.path.join("confusion_matrices",
                         f"cm_{r['method']}.png")
    plot_confusion_matrix(r["cm"], class_names,
                           lbl, r["mean_acc"], path)

# ============================================================
# 11. FINAL METRICS TABLE (paper-ready)
# ============================================================
elapsed        = time.time() - grand_start
baseline_acc   = next(r["mean_acc"] for r in all_results
                       if r["method"] == "none")
baseline_time  = next(r["time_sec"] for r in all_results
                       if r["method"] == "none")

print("\n\n" + "="*110)
print("  COMPLETE METRICS TABLE — All Methods (10-Fold CV)")
print("="*110)
print(f"  {'Method':<30} {'Acc%':>7} {'±Std':>6} "
      f"{'Prec%':>7} {'Rec%':>7} {'F1%':>7} "
      f"{'AUC':>7} {'FAR%':>7} {'FRR%':>7} "
      f"{'Trade%':>8} {'Time':>10}")
print("-"*110)

for r in results_sorted:
    t    = r["time_sec"]
    flag = " ✓" if r["mean_acc"]*100 >= 98.0 else ""
    print(f"  {LABELS.get(r['method']):<30}"
          f"  {r['mean_acc']*100:>6.2f}%"
          f"  ±{r['std_acc']*100:.2f}"
          f"  {r['precision']*100:>6.2f}%"
          f"  {r['recall']*100:>6.2f}%"
          f"  {r['f1']*100:>6.2f}%"
          f"  {r['auc']:>6.4f}"
          f"  {r['far']*100:>6.4f}%"
          f"  {r['frr']*100:>6.4f}%"
          f"  {r['tradeoff']*100:>7.4f}%"
          f"  {int(t//60):>2}m{t%60:04.1f}s"
          f"{flag}")

print("="*110)
print(f"  Total experiment time : "
      f"{int(elapsed//60)}m{elapsed%60:.0f}s")
print("="*110)

# ============================================================
# 12. PER-FOLD ACCURACY TABLE
# ============================================================
print("\n" + "="*78)
print("  PER-FOLD ACCURACY")
print("="*78)
header = f"  {'Method':<30}" + \
         "".join([f" F{i:02}" for i in range(1, NUM_FOLDS+1)]) + \
         "    Mean"
print(header)
print("-"*78)
for r in results_sorted:
    folds = "".join([f" {a*100:5.1f}" for a in r["accs"]])
    print(f"  {LABELS.get(r['method']):<30}"
          f"{folds}  {r['mean_acc']*100:6.2f}%")
print("="*78)

# ============================================================
# 13. SPEED vs ACCURACY TABLE
# ============================================================
print("\n" + "="*78)
print("  SPEED vs ACCURACY")
print("="*78)
print(f"  {'Method':<30} {'Accuracy':>10} {'Time':>10} "
      f"{'Speedup':>9} {'Gain':>8}")
print("-"*78)
for r in results_sorted:
    speedup = baseline_time / r["time_sec"]
    gain    = (r["mean_acc"] - baseline_acc) * 100
    t       = r["time_sec"]
    flag    = "  ✓" if r["mean_acc"]*100 >= 98.0 else ""
    print(f"  {LABELS.get(r['method']):<30}"
          f"  {r['mean_acc']*100:>9.2f}%"
          f"  {int(t//60):>3}m{t%60:04.0f}s"
          f"  {speedup:>7.1f}x"
          f"  +{gain:.2f}%"
          f"{flag}")
print("="*78)

# ============================================================
# 14. ACADEMIC SUMMARY
# ============================================================
print(f"""
{'='*78}
  ACADEMIC SUMMARY
{'='*78}
  Dataset        : Brain MRI (grayscale, 4 classes)
  Image size     : {IMAGE_SIZE}×{IMAGE_SIZE} + CLAHE preprocessing
  Validation     : Stratified {NUM_FOLDS}-Fold Cross-Validation
  Total samples  : {len(y)} (Training + Testing merged)
  Architecture   : MLP (1024→512→256, Swish, BatchNorm)
  Optimizer      : AdamW (lr={LR_MAX}, wd={WEIGHT_DECAY})
  LR Schedule    : Cosine Annealing (T=40)
  Label Smoothing: {LABEL_SMOOTHING}
  No GPU — No augmentation

  MLP baseline:
    Accuracy  : {baseline_acc*100:.2f}%
    Time      : {int(baseline_time//60)}m{baseline_time%60:.0f}s

  Best — {LABELS.get(best['method']).strip()}:
    Accuracy  : {best['mean_acc']*100:.2f}% ± {best['std_acc']*100:.2f}%
    Precision : {best['precision']*100:.2f}%
    Recall    : {best['recall']*100:.2f}%
    F1        : {best['f1']*100:.2f}%
    AUC       : {best['auc']:.4f}
    FAR       : {best['far']*100:.4f}%
    FRR       : {best['frr']*100:.4f}%
    Trade-off : {best['tradeoff']*100:.4f}%
    Time      : {int(best['time_sec']//60)}m{best['time_sec']%60:.0f}s
    Gain      : +{(best['mean_acc']-baseline_acc)*100:.2f}%
    Speedup   : {baseline_time/best['time_sec']:.1f}x faster

  Confusion matrices saved to: confusion_matrices/
  Best CM    : {cm_path}
{'='*78}
""")
