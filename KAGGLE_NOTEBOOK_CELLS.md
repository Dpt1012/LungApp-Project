# Kaggle notebook cells de debug dataset `data/masks`

Copy tung cell vao Kaggle Notebook. Chay Cell 1-3 truoc; neu Cell 3 in duoc sample pairs thi moi chay file train day du.

## Cell 1 - Imports va path

```python
import os
from pathlib import Path
import numpy as np

INPUT_ROOT = Path("/kaggle/input")

def print_tree(path, max_depth=4, max_items=80):
    path = Path(path)
    base_depth = len(path.parts)
    count = 0
    for item in sorted(path.rglob("*")):
        depth = len(item.parts) - base_depth
        if depth > max_depth:
            continue
        indent = "  " * (depth - 1)
        suffix = "/" if item.is_dir() else ""
        print(f"{indent}{item.name}{suffix}")
        count += 1
        if count >= max_items:
            print("...")
            break

print_tree(INPUT_ROOT / "lung-cancer-segment", max_depth=5)
```

## Cell 2 - Tim folder train/val

```python
def find_preprocessed_dirs(root):
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_dir():
            continue
        train_dir = candidate / "train"
        val_dir = candidate / "val"
        if train_dir.exists() and val_dir.exists():
            if list(train_dir.rglob("*.npy")) and list(val_dir.rglob("*.npy")):
                return candidate, train_dir, val_dir
    raise FileNotFoundError("Khong tim thay train/val co file .npy")

DATASET_ROOT, TRAIN_DIR, VAL_DIR = find_preprocessed_dirs(INPUT_ROOT)
print("DATASET_ROOT =", DATASET_ROOT)
print("TRAIN_DIR    =", TRAIN_DIR)
print("VAL_DIR      =", VAL_DIR)
```

## Cell 3 - Test pair `data/*.npy` voi `masks/*.npy`

```python
def find_npy_pairs(split_dir):
    pairs = []
    seen = set()

    def numeric_stem(path):
        try:
            return int(path.stem)
        except ValueError:
            return path.stem

    for case_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        data_dir = case_dir / "data"
        masks_dir = case_dir / "masks"
        if not data_dir.exists() or not masks_dir.exists():
            continue

        data_files = sorted(data_dir.rglob("*.npy"), key=numeric_stem)
        mask_files = sorted(masks_dir.rglob("*.npy"), key=numeric_stem)
        if not data_files or not mask_files:
            continue

        data_by_rel = {p.relative_to(data_dir).with_suffix("").as_posix(): p for p in data_files}
        data_by_stem = {p.stem: p for p in data_files}
        case_pairs = []

        for mask_path in mask_files:
            rel_key = mask_path.relative_to(masks_dir).with_suffix("").as_posix()
            image_path = data_by_rel.get(rel_key) or data_by_stem.get(mask_path.stem)
            if image_path is not None:
                case_pairs.append((image_path, mask_path, case_dir.name))

        if not case_pairs and len(data_files) == len(mask_files):
            case_pairs = [(image_path, mask_path, case_dir.name)
                          for image_path, mask_path in zip(data_files, mask_files)]

        for image_path, mask_path, case_id in case_pairs:
            key = (str(image_path), str(mask_path))
            if key not in seen:
                pairs.append((image_path, mask_path, case_id))
                seen.add(key)

    return pairs

train_pairs = find_npy_pairs(TRAIN_DIR)
val_pairs = find_npy_pairs(VAL_DIR)
print("train pairs:", len(train_pairs))
print("val pairs:", len(val_pairs))
print("sample train:", train_pairs[:3])

img_path, mask_path, case_id = train_pairs[0]
img = np.load(img_path)
mask = np.load(mask_path)
print("case:", case_id)
print("image:", img_path, img.shape, img.dtype, img.min(), img.max())
print("mask :", mask_path, mask.shape, mask.dtype, mask.min(), mask.max(), "positive pixels:", (mask > 0).sum())

positive = None
for pair in train_pairs:
    img_path, mask_path, case_id = pair
    mask = np.load(mask_path)
    if (mask > 0).sum() > 0:
        positive = pair
        break

if positive:
    img_path, mask_path, case_id = positive
    img = np.load(img_path)
    mask = np.load(mask_path)
    print("first positive case:", case_id)
    print("positive image:", img_path, img.shape, img.dtype, img.min(), img.max())
    print("positive mask :", mask_path, mask.shape, mask.dtype, mask.min(), mask.max(), "positive pixels:", (mask > 0).sum())
else:
    print("No positive mask found in train pairs")
```

## Cell 4 - Neu Cell 3 OK

Dung file da sua:

```text
kaggle_lung_tumor_segmentation_unet.py
```

Copy toan bo file do vao Kaggle Notebook va chay. Ban moi da co fix `train/0/data` + `train/0/masks`.

Khi chay dung, notebook se in kieu:

```text
Dataset kind: preprocessed_npy
train: found ... image/mask pairs
  sample pair case=0: ... -> ...
```
