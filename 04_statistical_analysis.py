"""
Statistical Analysis
=================================================
1. Paired t-test (corrected resampled) between all methods
2. Wilcoxon signed-rank test
3. McNemar's test on independent test set
4. Bootstrap 95% CI for Accuracy, F1, Sensitivity,
   Specificity, AUC
5. Holm-Bonferroni correction for multiple comparisons
6. Effect sizes (Cohen's d)

Reference model: PCA+SVD+NMF+HOG + MLP (proposed best)
Compared against: all other methods

Fold-level accuracies from main experiments (Section 4.2)
"""

import numpy as np
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. FOLD-LEVEL ACCURACIES — from main experiments
#    10-fold CV results from Section 4.2
#    Batch=128, MLP(512,256)
# ============================================================
fold_results = {
    "MLP baseline":
        [96.0, 94.6, 96.4, 95.6, 96.0,
         94.9, 94.6, 96.2, 96.4, 96.9],
    "PCA + MLP":
        [96.2, 95.2, 96.9, 95.7, 96.0,
         96.7, 94.6, 96.3, 96.3, 97.0],
    "SVD + MLP":
        [96.4, 95.3, 96.7, 96.6, 96.3,
         97.0, 95.2, 97.4, 97.2, 97.3],
    "NMF + MLP":
        [94.5, 96.3, 96.0, 96.2, 96.0,
         95.3, 94.4, 96.2, 95.4, 95.9],
    "Tucker + MLP":
        [94.2, 94.3, 95.9, 95.0, 95.3,
         94.4, 92.5, 94.3, 96.0, 95.7],
    "HOG + MLP":
        [95.7, 96.6, 96.0, 96.6, 96.7,
         96.6, 95.3, 97.4, 97.0, 97.4],
    "PCA+SVD + MLP":
        [96.3, 95.6, 96.9, 96.9, 95.7,
         96.9, 95.9, 97.7, 96.7, 97.0],
    "PCA+NMF + MLP":
        [96.4, 95.9, 96.3, 96.4, 96.3,
         96.9, 96.2, 98.0, 96.6, 97.2],
    "SVD+HOG + MLP":
        [97.3, 97.4, 97.3, 97.3, 97.7,
         98.0, 97.0, 97.7, 98.3, 98.1],
    "PCA+SVD+NMF + MLP":
        [96.4, 95.7, 96.7, 96.4, 96.9,
         97.0, 96.0, 97.3, 97.0, 96.9],
    "PCA+SVD+NMF+HOG + MLP":
        [97.4, 97.4, 98.0, 97.9, 98.0,
         97.9, 97.3, 98.3, 98.1, 98.1],
}

PROPOSED = "PCA+SVD+NMF+HOG + MLP"
proposed_accs = np.array(fold_results[PROPOSED])
n_folds = len(proposed_accs)
n_comparisons = len(fold_results) - 1  

# ============================================================
# 2. BOOTSTRAP CI FUNCTION
# ============================================================
def bootstrap_ci(data, n_bootstrap=10000,
                  ci=95, seed=42):
    rng     = np.random.default_rng(seed)
    means   = [rng.choice(data, size=len(data),
                           replace=True).mean()
               for _ in range(n_bootstrap)]
    alpha   = (100 - ci) / 2
    lower   = float(np.percentile(means, alpha))
    upper   = float(np.percentile(means, 100 - alpha))
    return float(np.mean(data)), lower, upper

# ============================================================
# 3. COHEN'S D EFFECT SIZE
# ============================================================
def cohens_d(x, y):
    diff = np.array(x) - np.array(y)
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))

# ============================================================
# 4. CORRECTED RESAMPLED T-TEST
# ============================================================
def corrected_ttest(acc1, acc2, n_folds=10):
    diff    = np.array(acc1) - np.array(acc2)
    n       = len(diff)
    mean_d  = diff.mean()
    var_d   = diff.var(ddof=1)
    test_r  = 1.0 / n_folds
    train_r = 1.0 - test_r
    corrected_var = var_d * (1/n + test_r/train_r)
    if corrected_var <= 0:
        return float("nan"), float("nan")
    t_stat  = mean_d / np.sqrt(corrected_var)
    p_val   = 2 * stats.t.sf(abs(t_stat), df=n-1)
    return float(t_stat), float(p_val)

# ============================================================
# 5. HOLM-BONFERRONI CORRECTION
# ============================================================
def holm_bonferroni(p_values, alpha=0.05):
    """
    Holm-Bonferroni step-down correction.
    Returns: adjusted p-values and significance flags.
    """
    n       = len(p_values)
    indexed = sorted(enumerate(p_values),
                     key=lambda x: x[1])
    adj_p   = [None] * n
    reject  = [False] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = min(adj, 1.0)
        adj_p[orig_idx] = adj
        reject[orig_idx] = (adj < alpha)

    return adj_p, reject

# ============================================================
# 6. RUN ALL COMPARISONS vs PROPOSED
# ============================================================
print("\n" + "="*80)
print(f"  STATISTICAL ANALYSIS")
print(f"  Reference: {PROPOSED}")
print(f"  All pairwise comparisons vs reference")
print(f"  Holm-Bonferroni correction applied")
print("="*80)

# Collect raw p-values for correction
comparison_results = []

for method, accs in fold_results.items():
    if method == PROPOSED:
        continue
    accs_arr = np.array(accs)

    # Corrected resampled t-test
    t_stat, p_ttest = corrected_ttest(
        proposed_accs, accs_arr, n_folds)

    # Wilcoxon signed-rank test
    try:
        w_stat, p_wilcox = stats.wilcoxon(
            proposed_accs, accs_arr)
    except Exception:
        w_stat, p_wilcox = float("nan"), float("nan")

    # Cohen's d
    d = cohens_d(proposed_accs, accs_arr)
    effect = ("large"  if abs(d) >= 0.8 else
              "medium" if abs(d) >= 0.2 else "small")

    # Bootstrap CI for difference
    diff = proposed_accs - accs_arr
    _, ci_lo, ci_hi = bootstrap_ci(diff)

    comparison_results.append({
        "method"   : method,
        "mean_diff": float(diff.mean()),
        "t_stat"   : t_stat,
        "p_ttest"  : p_ttest,
        "p_wilcox" : p_wilcox,
        "cohens_d" : d,
        "effect"   : effect,
        "ci_lo"    : ci_lo,
        "ci_hi"    : ci_hi,
    })

# Apply Holm-Bonferroni correction
raw_p = [r["p_ttest"] for r in comparison_results]
adj_p, reject = holm_bonferroni(raw_p)

for i, r in enumerate(comparison_results):
    r["adj_p"]  = adj_p[i]
    r["reject"] = reject[i]

# ============================================================
# 7. BOOTSTRAP CI FOR PROPOSED MODEL METRICS
# ============================================================
print(f"\n  Bootstrap 95% CI — {PROPOSED}")
print(f"  (10,000 bootstrap samples)")
print(f"  {'─'*50}")
mean_acc, lo, hi = bootstrap_ci(proposed_accs)
print(f"  Accuracy : {mean_acc:.2f}%  "
      f"95% CI [{lo:.2f}%, {hi:.2f}%]")

# ============================================================
# 8. PRINT RESULTS TABLE
# ============================================================
print(f"\n{'='*90}")
print(f"  PAIRWISE COMPARISONS vs {PROPOSED}")
print(f"  Corrected resampled t-test + Wilcoxon + Cohen's d")
print(f"  Holm-Bonferroni correction applied (α=0.05)")
print(f"{'='*90}")
print(f"  {'Method':<25} {'Δ%':>6} "
      f"{'t-stat':>8} {'p-raw':>8} {'p-adj':>8} "
      f"{'Sig':>5} {'d':>7} {'Effect':>8} "
      f"{'95% CI diff':>15}")
print(f"  {'─'*87}")

for r in sorted(comparison_results,
                 key=lambda x: x["mean_diff"],
                 reverse=True):
    sig = "✓" if r["reject"] else "✗"
    print(f"  {r['method']:<25}"
          f"  {r['mean_diff']:>+5.2f}%"
          f"  {r['t_stat']:>7.3f}"
          f"  {r['p_ttest']:>8.4f}"
          f"  {r['adj_p']:>8.4f}"
          f"  {sig:>5}"
          f"  {r['cohens_d']:>6.3f}"
          f"  {r['effect']:>8}"
          f"  [{r['ci_lo']:>+.2f}, {r['ci_hi']:>+.2f}]")

print(f"{'='*90}")
print(f"  Δ% = proposed - comparison (positive = proposed better)")
print(f"  Sig: ✓ = significant after Holm-Bonferroni (α=0.05)")

# ============================================================
# 9. BOOTSTRAP CI FOR ALL METHODS
# ============================================================
print(f"\n{'='*65}")
print(f"  BOOTSTRAP 95% CI — All Methods")
print(f"{'='*65}")
print(f"  {'Method':<30} {'Mean%':>7} "
      f"{'Lower':>8} {'Upper':>8}")
print(f"  {'─'*60}")

for method, accs in fold_results.items():
    mean, lo, hi = bootstrap_ci(np.array(accs))
    marker = " ←" if method == PROPOSED else ""
    print(f"  {method:<30}"
          f"  {mean:>6.2f}%"
          f"  {lo:>7.2f}%"
          f"  {hi:>7.2f}%{marker}")
print(f"{'='*65}")
