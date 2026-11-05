# Brain Tumor MRI Classification
### Hybrid Dimensionality Reduction and MLP-Based Framework

This repository contains the source code for the paper:

> **"Brain Tumor MRI Classification Using a Hybrid PCA, SVD, NMF, and HOG Feature Extraction Framework with MLP"**
> May Al-Saidi, Karam Sallam — University of Sharjah

---

## Dataset

**Brain Tumor MRI Dataset** (Kaggle)
- 4 classes: Glioma, Meningioma, Pituitary Tumour, No Tumour
- Training: 5,712 images | Testing: 1,311 images | Total: 7,023 images
- Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Place the dataset in a folder named `Images/` with the following structure:

Images/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/


---

## Requirements


Python        3.10.11
TensorFlow    2.21.0
scikit-learn  1.7.2
NumPy         2.2.6
OpenCV        4.x
imutils
scipy


Install dependencies:
```bash
pip install tensorflow scikit-learn numpy opencv-python imutils scipy
pip install tensorly  # required for 10_decomposition_visualization.py only
```

---

## Repository Structure

| File | Description |
|------|-------------|
| `01_decomposition_benchmark.py` | Main experiment — compares 11 feature extraction configurations (PCA, SVD, NMF, Tucker, HOG and combinations) using 10-fold cross-validation on the full 7,023-image dataset |
| `02_cnn_fair_comparison.py` | Fair comparison between the proposed MLP and CNN baselines (MLP on pixels, MLP on features, Simple CNN, Fusion CNN+HOG) under identical experimental conditions |
| `03_ablation_study.py` | Leave-one-feature-out ablation study — evaluates the contribution of each component in the hybrid PCA+SVD+NMF+HOG framework |
| `04_statistical_analysis.py` | Statistical significance testing — corrected resampled t-test, Holm–Bonferroni correction, Cohen's d, and Bootstrap 95% confidence intervals |
| `05_metrics_final.py` | Final model evaluation on the independent test set (1,311 images) following the correct train/test protocol — reports Accuracy, F1, AUC, FAR, FRR |
| `06_inference_time_measurement.py` | Per-image inference latency decomposed into three stages: preprocessing, feature extraction, and classification (100 runs, 10 warm-up passes) |
| `07_far_frr_binary_collapse.py` | FAR and FRR computation via binary collapse (Tumour vs. No Tumour) with per-class metrics |
| `08_dataset_duplication_check.py` | Dataset integrity check using MD5 hashing (exact duplicates) and perceptual hashing (near-duplicates) across training and test splits |
| `09_hyperparameter_tuning.py` | MLP hyperparameter tuning — image size, batch size, activation function, and learning rate selection on the development set (5,712 images) |
| `10_decomposition_visualization.py` | Generates Figure 2 — qualitative visualization of all decomposition methods applied to a sample brain MRI image (requires tensorly) |

---

## Experimental Configuration

| Setting | Value |
|---------|-------|
| Image size | 64×64 (grayscale + CLAHE) |
| Cross-validation | Stratified 10-Fold |
| Batch size | 128 |
| Optimizer | AdamW (lr=0.001, wd=1e-4) |
| LR schedule | Cosine Annealing (T=40) |
| Activation | Swish |
| Dropout | 0.25 |
| Label smoothing | 0.02 |
| Random seed | 42 |
| Hardware | Intel Core i7-1065G7, 16 GB RAM (CPU only) |

---

## Key Results

**Development phase** (7,023 images, 10-fold CV):

| Method | Accuracy |
|--------|----------|
| PCA + SVD + NMF + HOG + MLP | **97.85% ± 0.33%** |
| SVD + HOG + MLP | 97.49% ± 0.41% |
| PCA + SVD + NMF + MLP | 96.78% ± 0.52% |

**Final evaluation** (5,712 dev / 1,311 test split):

| Metric | Value |
|--------|-------|
| CV Accuracy (dev set) | 96.20% ± 0.63% |
| Test Accuracy | 95.96% |
| Macro F1 | 95.70% |
| AUC | 0.9946 |
| FAR | 0.0000% |
| FRR | 0.7726% |

---

## Reproducibility Note

A fixed random seed (42) is used throughout. Due to non-deterministic
floating-point operations in TensorFlow on CPU hardware, results may
vary slightly across runs (typically within ±0.5%).

---

## Contact

For questions or to request the trained model weights, please contact:
**U25102653@sharjah.ac.ae**
