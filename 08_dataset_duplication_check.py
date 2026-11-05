"""
Dataset Duplication Check
==========================
  1. MD5 hash  → exact duplicates
  2. Perceptual hash → near-duplicates (resized/altered)
  3. Train/Test overlap check
  4. Patient ID → not available (Kaggle dataset)
"""
import os
import cv2
import hashlib
import numpy as np
from imutils import paths
from collections import defaultdict

# ============================================================
# SETTINGS
# ============================================================
BASE_PATH  = "Images"
TRAIN_DIR  = os.path.join(BASE_PATH, "Training")
TEST_DIR   = os.path.join(BASE_PATH, "Testing")
IMAGE_SIZE = 64

# ============================================================
# 1. MD5 HASH — exact duplicates
# ============================================================
def get_md5(image_path):
    """Compute MD5 hash of raw image file bytes."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ============================================================
# 2. PERCEPTUAL HASH (pHash)
#    Resize to 8x8 → DCT → compare mean
#    Detects resized or slightly altered duplicates
# ============================================================
def get_phash(image_path, hash_size=8):
    """
    Compute perceptual hash (pHash) of an image.
    Two images with Hamming distance <= 5 are near-duplicates.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img    = cv2.resize(img, (hash_size * 4, hash_size * 4))
    # DCT-based hash
    img_f  = np.float32(img)
    dct    = cv2.dct(img_f)
    dct_low = dct[:hash_size, :hash_size]
    mean   = dct_low.mean()
    return "".join(["1" if v > mean else "0"
                    for v in dct_low.flatten()])

def hamming(hash1, hash2):
    """Hamming distance between two binary hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

# ============================================================
# MAIN
# ============================================================
print("\n" + "="*60)
print("  DATASET DUPLICATION CHECK")
print("="*60)

# Collect all image paths with their split label
all_paths = []
for img_path in paths.list_images(TRAIN_DIR):
    all_paths.append((img_path, "train"))
for img_path in paths.list_images(TEST_DIR):
    all_paths.append((img_path, "test"))

print(f"\n  Total images : {len(all_paths)}")
print(f"  Training     : {sum(1 for _,s in all_paths if s=='train')}")
print(f"  Testing      : {sum(1 for _,s in all_paths if s=='test')}")

# ── 1. MD5 exact duplicates ──────────────────────────────────
print("\n  [1] Checking exact duplicates (MD5)...")
md5_map = defaultdict(list)
for img_path, split in all_paths:
    h = get_md5(img_path)
    md5_map[h].append((img_path, split))

exact_dups  = {h: v for h, v in md5_map.items() if len(v) > 1}
train_test_exact = []
for h, items in exact_dups.items():
    splits = [s for _, s in items]
    if "train" in splits and "test" in splits:
        train_test_exact.append(items)

print(f"  Exact duplicate groups   : {len(exact_dups)}")
print(f"  Train-Test exact overlap : {len(train_test_exact)}")
if exact_dups:
    print("  !! Duplicate files found:")
    for h, items in list(exact_dups.items())[:5]:
        for path, split in items:
            print(f"     [{split}] {path}")
else:
    print("  ✓ No exact duplicates found")

# ── 2. Perceptual hash near-duplicates ───────────────────────
print("\n  [2] Checking near-duplicates (pHash, threshold=5)...")
phash_list = []
for img_path, split in all_paths:
    ph = get_phash(img_path)
    if ph:
        phash_list.append((img_path, split, ph))

near_dups       = 0
train_test_near = 0
# Compare all pairs (sample check for large datasets)
# Full O(n²) check
checked = 0
for i in range(len(phash_list)):
    for j in range(i+1, len(phash_list)):
        dist = hamming(phash_list[i][2], phash_list[j][2])
        if dist <= 5:
            near_dups += 1
            if (phash_list[i][1] != phash_list[j][1]):
                train_test_near += 1
                if train_test_near <= 3:
                    print(f"  !! Near-duplicate across splits:")
                    print(f"     train: {phash_list[i][0]}")
                    print(f"     test : {phash_list[j][0]}")
                    print(f"     Hamming distance: {dist}")

print(f"  Near-duplicate pairs     : {near_dups}")
print(f"  Train-Test near overlap  : {train_test_near}")
if train_test_near == 0:
    print("  ✓ No train-test near-duplicates found")

# ── 3. Patient ID ────────────────────────────────────────────
print("\n  [3] Patient ID check...")
print("  Dataset: Kaggle Brain Tumor MRI Dataset")
print("  Patient IDs: NOT available in this dataset")
print("  → StratifiedGroupKFold cannot be applied")
print("  → This is acknowledged as a limitation")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY FOR PAPER")
print("="*60)
print(f"""
  Exact duplicates (MD5)     : {len(exact_dups)} groups
  Train-Test exact overlap   : {len(train_test_exact)} pairs
  Near-duplicates (pHash≤5)  : {near_dups} pairs
  Train-Test near overlap    : {train_test_near} pairs
  Patient IDs available      : No

  """
print("="*60)
