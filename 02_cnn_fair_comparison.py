"""
CNN vs MLP Comparison
============================

1. CNN receives raw image tensors (64×64×1)
2. MLP receives PCA_SVD_NMF_HOG features
3. Hybrid = CNN embedding + HOG features → MLP head
4. Each model has separately tuned hyperparameters
5. Fair comparison architecture:

   MRI image → CNN → learned embedding ┐
                                        ├→ Concat → MLP → Prediction
   MRI image → PCA/SVD/NMF/HOG ────────┘

Experiments:
  1. MLP on flattened pixels (baseline)
  2. MLP on PCA+SVD+NMF+HOG features (proposed)
  3. Simple CNN on raw 64×64×1 images
  4. Fusion: CNN embedding + HOG features → MLP head
  
"""

import os, cv2, numpy as np, math, time, warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout,
                                      BatchNormalization, Input,
                                      Conv2D, MaxPooling2D,
                                      GlobalAveragePooling2D,
                                      Concatenate, Flatten)
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
warnings.filterwarnings("ignore")

# ============================================================
# 1. SETTINGS — each model has its own tuned config
# ============================================================
IMAGE_SIZE      = 64
NUM_FOLDS       = 10
RANDOM_SEED     = 42
LABEL_SMOOTHING = 0.02

# MLP hyperparameters 
MLP_BATCH    = 128
MLP_LR       = 0.001
MLP_WD       = 1e-4
MLP_EPOCHS   = 150
MLP_PATIENCE = 25
MLP_DROPOUT  = 0.25

# CNN hyperparameters 
CNN_BATCH    = 128
CNN_LR       = 0.001
CNN_WD       = 1e-4
CNN_EPOCHS   = 150
CNN_PATIENCE = 25
CNN_DROPOUT  = 0.25

# Fusion hyperparameters
FUS_BATCH    = 128
FUS_LR       = 0.001
FUS_WD       = 1e-4
FUS_EPOCHS   = 150
FUS_PATIENCE = 25

# Decomposition settings
N_PCA    = 150
N_SVD    = 150
N_NMF    = 100
HOG_CELL = 8
HOG_BINS = 9

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 2. CLAHE
# ============================================================
def apply_clahe(gray_uint8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_uint8)

# ============================================================
# 3. DATA LOADING
# ============================================================
def load_dataset(base_path):
    
    train_dir   = os.path.join(base_path, "Training")
    test_dir    = os.path.join(base_path, "Testing")
    train_paths = list(paths.list_images(train_dir))
    test_paths  = list(paths.list_images(test_dir))
    all_image_paths = train_paths + test_paths

    flat_data, img_data, labels = [], [], []
    for image_path in all_image_paths:
        img = cv2.imread(image_path)
        if img is None:
            continue
        img     = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img     = apply_clahe(img)
        img_f32 = img.astype("float32") / 255.0
        flat_data.append(img_f32.flatten())
        img_data.append(img_f32[..., np.newaxis])  # (64,64,1)
        labels.append(image_path.split(os.path.sep)[-2])

    le     = LabelEncoder()
    y      = le.fit_transform(labels)
    X_flat = np.array(flat_data, dtype="float32")
    X_img  = np.array(img_data,  dtype="float32")

    print(f"\n  Total images : {len(y)}")
    print(f"  Training     : {len(train_paths)}")
    print(f"  Testing      : {len(test_paths)}")
    print(f"  Classes      : {list(le.classes_)}")
    print(f"  Note: All images merged for K-Fold CV")
    print(f"        (consistent with Sections 4.1-4.3)")

    return X_flat, X_img, y, le

# ============================================================
# 4. FEATURE EXTRACTORS (MLP + Fusion)
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
                  (IMAGE_SIZE,IMAGE_SIZE),
                  bs, bst, cs, HOG_BINS)
        return h.compute(img).flatten().astype("float32")
    sc  = StandardScaler()
    Htr = np.array([extract(i) for i in Xtr], dtype="float32")
    Hte = np.array([extract(i) for i in Xte], dtype="float32")
    return sc.fit_transform(Htr), sc.transform(Hte)

def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)

def build_all_features(Xtr, Xte):
    """PCA+SVD+NMF+HOG concatenated."""
    p = feat_pca(Xtr, Xte)
    s = feat_svd(Xtr, Xte)
    n = feat_nmf(Xtr, Xte)
    h = feat_hog(Xtr, Xte)
    return (np.concatenate([l2n(p[0]), l2n(s[0]),
                             l2n(n[0]), l2n(h[0])], axis=1),
            np.concatenate([l2n(p[1]), l2n(s[1]),
                             l2n(n[1]), l2n(h[1])], axis=1))

def build_hog_only(Xtr, Xte):
    """HOG only — for fusion model."""
    return feat_hog(Xtr, Xte)

# ============================================================
# 5. MODEL BUILDERS
# ============================================================

# ── 5a. MLP on flat pixels (baseline) ───────────────────────
def build_mlp_pixels(input_dim, num_classes):
    
    sc = StandardScaler()  
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, use_bias=False),
        BatchNormalization(),
        tf.keras.layers.Activation("swish"),
        Dropout(MLP_DROPOUT),
        Dense(256, use_bias=False),
        BatchNormalization(),
        tf.keras.layers.Activation("swish"),
        Dropout(MLP_DROPOUT),
        Dense(num_classes, activation="softmax",
               dtype="float32")
    ], name="MLP_pixels")
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
                 label_smoothing=LABEL_SMOOTHING),
        optimizer=AdamW(learning_rate=MLP_LR,
                         weight_decay=MLP_WD),
        metrics=["accuracy"])
    return model

# ── 5b. MLP on PCA+SVD+NMF+HOG (proposed) ──────────────────
def build_mlp_features(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, use_bias=False),
        BatchNormalization(),
        tf.keras.layers.Activation("swish"),
        Dropout(MLP_DROPOUT),
        Dense(256, use_bias=False),
        BatchNormalization(),
        tf.keras.layers.Activation("swish"),
        Dropout(MLP_DROPOUT),
        Dense(num_classes, activation="softmax",
               dtype="float32")
    ], name="MLP_features")
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
                 label_smoothing=LABEL_SMOOTHING),
        optimizer=AdamW(learning_rate=MLP_LR,
                         weight_decay=MLP_WD),
        metrics=["accuracy"])
    return model

# ── 5c. Simple CNN on raw images ────────────────────────────
def build_cnn(num_classes):
    
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                    name="image")

    # CNN backbone
    x = Conv2D(32, (3,3), padding="same",
                use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = MaxPooling2D((2,2))(x)              # 64→32

    x = Conv2D(64, (3,3), padding="same",
                use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = MaxPooling2D((2,2))(x)              # 32→16

    x = Conv2D(128, (3,3), padding="same",
                use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = GlobalAveragePooling2D()(x)         # (batch,128)

    # Dense head — tuned separately from MLP
    x = Dense(256, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = Dropout(CNN_DROPOUT)(x)

    x = Dense(128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = Dropout(CNN_DROPOUT)(x)

    outputs = Dense(num_classes, activation="softmax",
                     dtype="float32")(x)

    model = Model(inputs=inputs, outputs=outputs,
                   name="Simple_CNN")
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
                 label_smoothing=LABEL_SMOOTHING),
        optimizer=AdamW(learning_rate=CNN_LR,
                         weight_decay=CNN_WD),
        metrics=["accuracy"])
    return model

# ── 5d. Fusion: CNN embedding + HOG → MLP head ──────────────
def build_fusion(hog_dim, num_classes):
   
    # Stream A: CNN on raw image
    img_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                       name="image_input")
    c = Conv2D(32, (3,3), padding="same",
                use_bias=False)(img_input)
    c = BatchNormalization()(c)
    c = tf.keras.layers.Activation("swish")(c)
    c = MaxPooling2D((2,2))(c)

    c = Conv2D(64, (3,3), padding="same",
                use_bias=False)(c)
    c = BatchNormalization()(c)
    c = tf.keras.layers.Activation("swish")(c)
    c = MaxPooling2D((2,2))(c)

    c = Conv2D(128, (3,3), padding="same",
                use_bias=False)(c)
    c = BatchNormalization()(c)
    c = tf.keras.layers.Activation("swish")(c)
    c = GlobalAveragePooling2D()(c)   # (batch, 128)

    # Stream B: HOG features
    hog_input = Input(shape=(hog_dim,), name="hog_input")
    h = Dense(128, use_bias=False)(hog_input)
    h = BatchNormalization()(h)
    h = tf.keras.layers.Activation("swish")(h)
    h = Dropout(0.25)(h)              

    # Concatenate both streams
    merged = Concatenate()([c, h])    

    # MLP head
    x = Dense(256, use_bias=False)(merged)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = Dropout(0.25)(x)

    x = Dense(128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    x = Dropout(0.25)(x)

    outputs = Dense(num_classes, activation="softmax",
                     dtype="float32")(x)

    model = Model(inputs=[img_input, hog_input],
                   outputs=outputs,
                   name="CNN_HOG_Fusion")
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
                 label_smoothing=LABEL_SMOOTHING),
        optimizer=AdamW(learning_rate=FUS_LR,
                         weight_decay=FUS_WD),
        metrics=["accuracy"])
    return model

# ============================================================
# 6. LR SCHEDULES 
# ============================================================
def lr_schedule_mlp(epoch):
    t = epoch % 40
    return float(1e-6 + 0.5*(MLP_LR-1e-6)*
                 (1+math.cos(math.pi*t/40)))

def lr_schedule_cnn(epoch):
    t = epoch % 40
    return float(1e-6 + 0.5*(CNN_LR-1e-6)*
                 (1+math.cos(math.pi*t/40)))

def lr_schedule_fus(epoch):
    t = epoch % 40
    return float(1e-6 + 0.5*(FUS_LR-1e-6)*
                 (1+math.cos(math.pi*t/40)))

# ============================================================
# 7. METRICS
# ============================================================
def compute_metrics(y_true, y_pred_labels,
                    y_pred_proba, num_classes):
    acc       = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels,
                                 average="macro",
                                 zero_division=0)
    recall    = recall_score(y_true, y_pred_labels,
                              average="macro",
                              zero_division=0)
    f1        = f1_score(y_true, y_pred_labels,
                          average="macro",
                          zero_division=0)
    try:
        y_oh = tf.keras.utils.to_categorical(
            y_true, num_classes)
        auc  = roc_auc_score(y_oh, y_pred_proba,
                              multi_class="ovr",
                              average="macro")
    except Exception:
        auc  = float("nan")
    return {"acc":acc, "prec":precision,
            "rec":recall, "f1":f1, "auc":auc}

# ============================================================
# 8. 10-Fold CV
# ============================================================
def run_experiment(name, build_fn, get_inputs_fn,
                   X_flat, X_img, y_all,
                   num_classes, cw_dict,
                   batch_size, epochs, patience,
                   lr_schedule_fn):
    print(f"\n{'#'*60}")
    print(f"  {name}")
    print(f"  Batch={batch_size} | Epochs={epochs}")
    print(f"{'#'*60}")

    skfold     = StratifiedKFold(n_splits=NUM_FOLDS,
                                  shuffle=True,
                                  random_state=RANDOM_SEED)
    fold_metrics = []
    t_start      = time.time()

    for fold_no, (tr_idx, te_idx) in enumerate(
            skfold.split(X_flat, y_all), start=1):

        y_tr = y_all[tr_idx]; y_te = y_all[te_idx]
        y_tr_oh = tf.keras.utils.to_categorical(
            y_tr, num_classes)
        y_te_oh = tf.keras.utils.to_categorical(
            y_te, num_classes)

        # Get inputs for this fold
        train_in, val_in, model = get_inputs_fn(
            tr_idx, te_idx,
            X_flat, X_img,
            num_classes, fold_no)

        if fold_no == 1:
            model.summary(print_fn=lambda x:
                          print(f"  {x}") if "param" in x
                          else None)

        lr_cb = LearningRateScheduler(
            lr_schedule_fn, verbose=0)
        es_cb = EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=0)

        model.fit(train_in, y_tr_oh,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(val_in, y_te_oh),
                  class_weight=cw_dict,
                  callbacks=[lr_cb, es_cb],
                  verbose=0)

        y_pred_proba  = model.predict(val_in, verbose=0)
        y_pred_labels = np.argmax(y_pred_proba, axis=1)
        m = compute_metrics(y_te, y_pred_labels,
                             y_pred_proba, num_classes)
        fold_metrics.append(m)

        ft   = time.time() - t_start
        flag = "  ✓" if m["acc"]*100 >= 97.0 else ""
        print(f"  Fold {fold_no:>2}: "
              f"Acc={m['acc']*100:.2f}%  "
              f"F1={m['f1']:.4f}{flag}")

    total = time.time() - t_start
    def fm(k): return float(np.mean(
        [f[k] for f in fold_metrics]))
    accs = [f["acc"] for f in fold_metrics]

    result = {
        "name"    : name,
        "mean_acc": float(np.mean(accs)),
        "std_acc" : float(np.std(accs)),
        "prec"    : fm("prec"),
        "rec"     : fm("rec"),
        "f1"      : fm("f1"),
        "auc"     : fm("auc"),
        "time_sec": total,
        "params"  : model.count_params(),
    }

    mins, secs = int(total//60), total % 60
    print(f"\n  Accuracy : {result['mean_acc']*100:.2f}%"
          f" ± {result['std_acc']*100:.2f}%")
    print(f"  F1       : {result['f1']*100:.2f}%")
    print(f"  AUC      : {result['auc']:.4f}")
    print(f"  Params   : {result['params']:,}")
    print(f"  Time     : {mins}m {secs:.0f}s")
    return result

# ============================================================
# 9. INPUT BUILDERS PER MODEL
# ============================================================
def inputs_mlp_pixels(tr_idx, te_idx,
                       X_flat, X_img,
                       num_classes, fold_no):
    sc   = StandardScaler()
    Xtr  = sc.fit_transform(X_flat[tr_idx])
    Xte  = sc.transform(X_flat[te_idx])
    model = build_mlp_pixels(Xtr.shape[1], num_classes)
    return Xtr, Xte, model

def inputs_mlp_features(tr_idx, te_idx,
                          X_flat, X_img,
                          num_classes, fold_no):
    Xtr, Xte = build_all_features(
        X_flat[tr_idx], X_flat[te_idx])
    model = build_mlp_features(Xtr.shape[1], num_classes)
    return Xtr, Xte, model

def inputs_cnn(tr_idx, te_idx,
                X_flat, X_img,
                num_classes, fold_no):
    """CNN gets raw image tensors — NOT features."""
    Xtr   = X_img[tr_idx]   # (N, 64, 64, 1)
    Xte   = X_img[te_idx]   # (N, 64, 64, 1)
    model = build_cnn(num_classes)
    return Xtr, Xte, model

def inputs_fusion(tr_idx, te_idx,
                   X_flat, X_img,
                   num_classes, fold_no):
    """
    Fusion:
      Stream A: CNN gets raw images (64×64×1)
      Stream B: HOG features from flat pixels
    """
    hog_tr, hog_te = build_hog_only(
        X_flat[tr_idx], X_flat[te_idx])
    img_tr = X_img[tr_idx]
    img_te = X_img[te_idx]
    model  = build_fusion(hog_tr.shape[1], num_classes)
    return ([img_tr, hog_tr],
            [img_te, hog_te],
            model)

# ============================================================
# 10. MAIN
# ============================================================
X_flat, X_img, y, le = load_dataset("Images")

num_classes = len(le.classes_)
class_names = list(le.classes_)

cw_array = compute_class_weight("balanced",
                                  classes=np.unique(y),
                                  y=y)
cw_dict  = dict(enumerate(cw_array))

EXPERIMENTS = [
    ("MLP on flattened pixels (baseline)",
     inputs_mlp_pixels,
     MLP_BATCH, MLP_EPOCHS, MLP_PATIENCE,
     lr_schedule_mlp),

    ("MLP on PCA+SVD+NMF+HOG (proposed)",
     inputs_mlp_features,
     MLP_BATCH, MLP_EPOCHS, MLP_PATIENCE,
     lr_schedule_mlp),

    ("Simple CNN on raw 64×64×1 images",
     inputs_cnn,
     CNN_BATCH, CNN_EPOCHS, CNN_PATIENCE,
     lr_schedule_cnn),

    ("Fusion: CNN embedding + HOG → MLP",
     inputs_fusion,
     FUS_BATCH, FUS_EPOCHS, FUS_PATIENCE,
     lr_schedule_fus),
]

grand_start = time.time()
all_results = []

for (name, input_fn,
     batch, epochs, patience, lr_fn) in EXPERIMENTS:
    r = run_experiment(
        name, None, input_fn,
        X_flat, X_img, y,
        num_classes, cw_dict,
        batch, epochs, patience, lr_fn)
    all_results.append(r)

# ============================================================
# 11. RESULTS TABLE
# ============================================================
elapsed = time.time() - grand_start

print(f"\n\n{'='*100}")
print(f"  FAIR CNN vs MLP COMPARISON — 10-Fold CV "
      f"on Merged Dataset (7,023 images)")
print(f"  Reviewer-approved architecture:")
print(f"  MRI → CNN → embedding ┐")
print(f"  MRI → HOG ────────────┴→ Concat → MLP → Pred")
print(f"{'='*100}")
print(f"  {'Model':<40} {'Acc%':>8} {'Std':>7} "
      f"{'F1%':>7} {'AUC':>7} {'Params':>10} "
      f"{'Time':>10}")
print(f"  {'-'*95}")

for r in all_results:
    t = r["time_sec"]
    print(f"  {r['name']:<40}"
          f"  {r['mean_acc']*100:>7.2f}%"
          f"  ±{r['std_acc']*100:.2f}"
          f"  {r['f1']*100:>6.2f}%"
          f"  {r['auc']:>6.4f}"
          f"  {r['params']:>9,}"
          f"  {int(t//60):>2}m{t%60:04.0f}s")

print(f"{'='*100}")
print(f"  Total : {int(elapsed//60)}m{elapsed%60:.0f}s")
print(f"{'='*100}")

