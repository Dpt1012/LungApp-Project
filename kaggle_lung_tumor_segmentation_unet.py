# %% [markdown]
# # Lung Tumor Segmentation on Kaggle
#
# Add this Kaggle input before running:
# - `rasoulisaeid/lung-cancer-segment`
#
# This notebook trains a lightweight 2D U-Net from 3D CT volumes and tumor masks.
# It exports:
# - `lung_tumor_unet_best.keras`
# - `lung_tumor_unet_best.h5`
# - `lung_tumor_unet_float32.tflite`
# - `preprocess_config.json`
# - `sample_predictions.png`

# %%
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def ensure_import(package_name, import_name=None):
    try:
        __import__(import_name or package_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


ensure_import("nibabel")
import nibabel as nib

import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# %% [markdown]
# ## Configuration

# %%
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
SLICE_DIR = WORK_DIR / "lung_tumor_2d_slices"

IMG_SIZE = (256, 256)
HU_MIN = -1000.0
HU_MAX = 400.0
VAL_SPLIT = 0.15

# Keep tumor slices plus nearby context slices, then add background slices for false-positive control.
MIN_MASK_PIXELS = 8
POSITIVE_CONTEXT_SLICES = 2
NEGATIVE_RATIO = 1.5

BATCH_SIZE = 16
EPOCHS = 60
BASE_FILTERS = 32

BEST_MODEL_PATH = WORK_DIR / "lung_tumor_unet_best.keras"
H5_MODEL_PATH = WORK_DIR / "lung_tumor_unet_best.h5"
TFLITE_MODEL_PATH = WORK_DIR / "lung_tumor_unet_float32.tflite"
CONFIG_PATH = WORK_DIR / "preprocess_config.json"
HISTORY_PATH = WORK_DIR / "training_history.csv"
PREDICTIONS_PNG = WORK_DIR / "sample_predictions.png"

# %% [markdown]
# ## Locate MSD Task06 Lung data
#
# Expected source folders:
# - `imagesTr/*.nii.gz`
# - `labelsTr/*.nii.gz`

# %%


def find_task06_dirs(root):
    image_dirs = sorted(root.rglob("imagesTr"))
    label_dirs = sorted(root.rglob("labelsTr"))
    candidates = []
    for image_dir in image_dirs:
        dataset_root = image_dir.parent
        label_dir = dataset_root / "labelsTr"
        if label_dir.exists():
            candidates.append((dataset_root, image_dir, label_dir))
    if not candidates and image_dirs and label_dirs:
        candidates.append((image_dirs[0].parent, image_dirs[0], label_dirs[0]))
    if not candidates:
        raise FileNotFoundError(
            "Could not find imagesTr/labelsTr under /kaggle/input. "
            "Add the Kaggle dataset rasoulisaeid/lung-cancer-segment first."
        )
    return candidates[0]


def find_preprocessed_dirs(root):
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_dir():
            continue
        train_dir = candidate / "train"
        val_dir = candidate / "val"
        if train_dir.exists() and val_dir.exists():
            if list(train_dir.rglob("*.npy")) and list(val_dir.rglob("*.npy")):
                return candidate, train_dir, val_dir
    raise FileNotFoundError(
        "Could not find preprocessed train/val folders with .npy files under /kaggle/input."
    )


try:
    DATASET_ROOT, IMAGES_TR, LABELS_TR = find_task06_dirs(KAGGLE_INPUT_ROOT)
    DATASET_KIND = "raw_nii"
    image_files = sorted(list(IMAGES_TR.glob("*.nii*")))
    label_map = {p.name: p for p in LABELS_TR.glob("*.nii*")}
except FileNotFoundError:
    DATASET_ROOT, PREPROCESSED_TRAIN, PREPROCESSED_VAL = find_preprocessed_dirs(KAGGLE_INPUT_ROOT)
    DATASET_KIND = "preprocessed_npy"
    image_files = []
    label_map = {}

if DATASET_KIND == "raw_nii":
    missing = [p.name for p in image_files if p.name not in label_map]
    if missing:
        raise FileNotFoundError(f"Missing labels for {len(missing)} images. First missing: {missing[0]}")

print("Dataset root:", DATASET_ROOT)
print("Dataset kind:", DATASET_KIND)
if DATASET_KIND == "raw_nii":
    print("Training volumes:", len(image_files))
else:
    print("Preprocessed train dir:", PREPROCESSED_TRAIN)
    print("Preprocessed val dir:", PREPROCESSED_VAL)

# %% [markdown]
# ## Convert 3D CT volumes to 2D slices
#
# The tumor target is small and class-imbalanced. We keep all positive slices, add nearby slices, and sample negatives.

# %%


def normalize_ct(slice_2d):
    slice_2d = np.clip(slice_2d, HU_MIN, HU_MAX)
    return (slice_2d - HU_MIN) / (HU_MAX - HU_MIN)


def resize_slice(slice_2d, method):
    tensor = tf.convert_to_tensor(slice_2d[..., np.newaxis], dtype=tf.float32)
    resized = tf.image.resize(tensor, IMG_SIZE, method=method).numpy()[..., 0]
    return resized


def choose_slice_axis(volume):
    # MSD Task06 is typically (H, W, Z). The slice axis is the smallest dimension in practice.
    return int(np.argmin(volume.shape))


def selected_slice_indices(mask_volume, rng):
    slice_has_tumor = mask_volume.reshape(-1, mask_volume.shape[-1]).sum(axis=0) > MIN_MASK_PIXELS
    positive = np.where(slice_has_tumor)[0].tolist()
    keep = set()
    for idx in positive:
        start = max(0, idx - POSITIVE_CONTEXT_SLICES)
        end = min(mask_volume.shape[-1], idx + POSITIVE_CONTEXT_SLICES + 1)
        keep.update(range(start, end))

    negative = [i for i in range(mask_volume.shape[-1]) if i not in keep]
    rng.shuffle(negative)
    negative_count = int(max(1, len(keep)) * NEGATIVE_RATIO)
    keep.update(negative[:negative_count])
    return sorted(keep)


def convert_volumes(volumes, split_name):
    rng = random.Random(SEED + (0 if split_name == "train" else 1000))
    image_out_dir = SLICE_DIR / split_name / "images"
    mask_out_dir = SLICE_DIR / split_name / "masks"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for volume_idx, image_path in enumerate(volumes):
        label_path = label_map[image_path.name]
        image_volume = nib.load(str(image_path)).get_fdata(dtype=np.float32)
        mask_volume = nib.load(str(label_path)).get_fdata(dtype=np.float32) > 0

        axis = choose_slice_axis(image_volume)
        image_volume = np.moveaxis(image_volume, axis, -1)
        mask_volume = np.moveaxis(mask_volume, axis, -1)

        for slice_idx in selected_slice_indices(mask_volume, rng):
            image_slice = normalize_ct(image_volume[..., slice_idx])
            mask_slice = mask_volume[..., slice_idx].astype(np.float32)

            image_slice = resize_slice(image_slice, method="bilinear")
            mask_slice = resize_slice(mask_slice, method="nearest")
            mask_slice = (mask_slice > 0.5).astype(np.uint8)

            stem = f"{image_path.stem.replace('.nii', '')}_z{slice_idx:04d}"
            image_out = image_out_dir / f"{stem}.npy"
            mask_out = mask_out_dir / f"{stem}.npy"
            np.save(image_out, (image_slice * 255).astype(np.uint8))
            np.save(mask_out, mask_slice)
            rows.append(
                {
                    "image": str(image_out),
                    "mask": str(mask_out),
                    "volume": image_path.name,
                    "slice": slice_idx,
                    "positive": int(mask_slice.sum() > 0),
                }
            )

        if (volume_idx + 1) % 5 == 0:
            print(f"{split_name}: converted {volume_idx + 1}/{len(volumes)} volumes")

    return pd.DataFrame(rows)


def is_mask_like(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        return True
    sample = arr.ravel()
    if sample.size > 300_000:
        sample = np.random.default_rng(SEED).choice(sample, size=300_000, replace=False)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return False
    unique = np.unique(sample)
    return len(unique) <= 8 and float(unique.min()) >= 0 and float(unique.max()) <= 5


def normalize_preprocessed_image(slice_2d):
    slice_2d = np.asarray(slice_2d, dtype=np.float32)
    mn = float(np.nanmin(slice_2d))
    mx = float(np.nanmax(slice_2d))
    if mn < 0:
        slice_2d = normalize_ct(slice_2d)
    elif mx > 1.5:
        divisor = 3071.0 if mx > 255.0 else 255.0
        slice_2d = slice_2d / divisor
    return np.clip(slice_2d, 0.0, 1.0)


def iter_slices_from_pair(image_arr, mask_arr):
    image_arr = np.squeeze(np.asarray(image_arr))
    mask_arr = np.squeeze(np.asarray(mask_arr))
    if image_arr.shape != mask_arr.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_arr.shape} vs {mask_arr.shape}")
    if image_arr.ndim == 2:
        yield image_arr, mask_arr
        return
    if image_arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image_arr.shape}")

    axis = choose_slice_axis(mask_arr)
    image_arr = np.moveaxis(image_arr, axis, -1)
    mask_arr = np.moveaxis(mask_arr, axis, -1)
    for slice_idx in range(image_arr.shape[-1]):
        yield image_arr[..., slice_idx], mask_arr[..., slice_idx]


def numeric_stem(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def find_npy_pairs(split_dir):
    pairs = []
    seen = set()

    # Kaggle rasoulisaeid/lung-cancer-segment layout:
    # train/0/data/*.npy and train/0/masks/*.npy, then val/57/data/*.npy, ...
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

    if pairs:
        return pairs

    for folder in sorted([p for p in split_dir.rglob("*") if p.is_dir()]):
        files = sorted(folder.glob("*.npy"))
        if len(files) < 2:
            continue

        loaded = []
        for file_path in files:
            try:
                arr = np.load(file_path, mmap_mode="r")
                if arr.ndim >= 2:
                    loaded.append((file_path, arr, is_mask_like(arr)))
            except Exception:
                continue

        mask_candidates = [item for item in loaded if item[2]]
        image_candidates = [item for item in loaded if not item[2]]
        if not mask_candidates or not image_candidates:
            continue

        mask_path, mask_arr, _ = mask_candidates[0]
        same_shape_images = [item for item in image_candidates if item[1].shape == mask_arr.shape]
        image_path, _, _ = same_shape_images[0] if same_shape_images else image_candidates[0]
        pairs.append((image_path, mask_path, folder.name))

    if pairs:
        return pairs

    all_files = sorted(split_dir.rglob("*.npy"), key=numeric_stem)
    masks = [p for p in all_files if any(token in p.name.lower() for token in ("mask", "label", "seg"))]
    images = [p for p in all_files if p not in masks]
    image_by_key = {}
    for image_path in images:
        key = image_path.stem.lower()
        for token in ("image", "img", "ct", "scan", "data"):
            key = key.replace(token, "")
        image_by_key[key] = image_path

    for mask_path in masks:
        key = mask_path.stem.lower()
        for token in ("mask", "label", "seg", "tumor"):
            key = key.replace(token, "")
        image_path = image_by_key.get(key)
        if image_path:
            pairs.append((image_path, mask_path, mask_path.parent.name))
    return pairs


def filter_slice_pairs(pairs, split_name):
    if split_name != "train":
        return pairs

    rng = random.Random(SEED)
    by_case = {}
    mask_sums = {}
    for image_path, mask_path, case_id in pairs:
        by_case.setdefault(case_id, []).append((image_path, mask_path, case_id))
        mask = np.load(mask_path, mmap_mode="r")
        mask_sums[str(mask_path)] = int((mask > 0).sum())

    selected = []
    for case_id, case_pairs in by_case.items():
        case_pairs = sorted(case_pairs, key=lambda item: numeric_stem(item[0]))
        positive_positions = [
            idx for idx, (_, mask_path, _) in enumerate(case_pairs)
            if mask_sums[str(mask_path)] >= MIN_MASK_PIXELS
        ]
        keep = set()
        for idx in positive_positions:
            start = max(0, idx - POSITIVE_CONTEXT_SLICES)
            end = min(len(case_pairs), idx + POSITIVE_CONTEXT_SLICES + 1)
            keep.update(range(start, end))

        negative_positions = [idx for idx in range(len(case_pairs)) if idx not in keep]
        rng.shuffle(negative_positions)
        negative_count = int(max(1, len(keep)) * NEGATIVE_RATIO)
        keep.update(negative_positions[:negative_count])
        selected.extend(case_pairs[idx] for idx in sorted(keep))

    print(f"{split_name}: selected {len(selected)}/{len(pairs)} slices after positive/context/negative sampling")
    return selected


def convert_preprocessed_split(split_dir, split_name):
    image_out_dir = SLICE_DIR / split_name / "images"
    mask_out_dir = SLICE_DIR / split_name / "masks"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_npy_pairs(split_dir)
    if not pairs:
        raise FileNotFoundError(f"No image/mask .npy pairs found in {split_dir}")
    print(f"{split_name}: found {len(pairs)} image/mask pairs")
    for sample_image, sample_mask, sample_case in pairs[:3]:
        print(f"  sample pair case={sample_case}: {sample_image.name} -> {sample_mask.name}")
    pairs = filter_slice_pairs(pairs, split_name)

    rows = []
    for pair_idx, (image_path, mask_path, case_id) in enumerate(pairs):
        image_arr = np.load(image_path)
        mask_arr = np.load(mask_path)
        for slice_idx, (image_slice, mask_slice) in enumerate(iter_slices_from_pair(image_arr, mask_arr)):
            image_slice = normalize_preprocessed_image(image_slice)
            mask_slice = (np.asarray(mask_slice) > 0).astype(np.float32)

            image_slice = resize_slice(image_slice, method="bilinear")
            mask_slice = resize_slice(mask_slice, method="nearest")
            mask_slice = (mask_slice > 0.5).astype(np.uint8)

            stem = f"{case_id}_{pair_idx:04d}_z{slice_idx:04d}"
            image_out = image_out_dir / f"{stem}.npy"
            mask_out = mask_out_dir / f"{stem}.npy"
            np.save(image_out, (image_slice * 255).astype(np.uint8))
            np.save(mask_out, mask_slice)
            rows.append(
                {
                    "image": str(image_out),
                    "mask": str(mask_out),
                    "volume": case_id,
                    "slice": slice_idx,
                    "positive": int(mask_slice.sum() > 0),
                }
            )
        if (pair_idx + 1) % 10 == 0:
            print(f"{split_name}: converted {pair_idx + 1}/{len(pairs)} npy pairs")
    return pd.DataFrame(rows)


if not SLICE_DIR.exists():
    if DATASET_KIND == "raw_nii":
        shuffled = image_files[:]
        random.Random(SEED).shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * VAL_SPLIT)))
        val_volumes = sorted(shuffled[:val_count])
        train_volumes = sorted(shuffled[val_count:])

        train_df = convert_volumes(train_volumes, "train")
        val_df = convert_volumes(val_volumes, "val")
    else:
        train_df = convert_preprocessed_split(PREPROCESSED_TRAIN, "train")
        val_df = convert_preprocessed_split(PREPROCESSED_VAL, "val")

    train_df.to_csv(SLICE_DIR / "train_manifest.csv", index=False)
    val_df.to_csv(SLICE_DIR / "val_manifest.csv", index=False)
else:
    train_df = pd.read_csv(SLICE_DIR / "train_manifest.csv")
    val_df = pd.read_csv(SLICE_DIR / "val_manifest.csv")

print("Train slices:", len(train_df), "positive:", int(train_df["positive"].sum()))
print("Val slices:", len(val_df), "positive:", int(val_df["positive"].sum()))
print("Slice dir:", SLICE_DIR)

# %% [markdown]
# ## Build TensorFlow datasets

# %%


def np_loader(image_path, mask_path):
    image_path = image_path.decode("utf-8")
    mask_path = mask_path.decode("utf-8")
    image = np.load(image_path).astype(np.float32) / 255.0
    mask = np.load(mask_path).astype(np.float32)
    return image[..., np.newaxis], mask[..., np.newaxis]


def tf_loader(image_path, mask_path):
    image, mask = tf.numpy_function(np_loader, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 1])
    mask.set_shape([IMG_SIZE[0], IMG_SIZE[1], 1])
    return image, mask


def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)
    image = tf.clip_by_value(image + tf.random.normal(tf.shape(image), stddev=0.015), 0.0, 1.0)
    return image, mask


def make_dataset(df, training):
    ds = tf.data.Dataset.from_tensor_slices((df["image"].values, df["mask"].values))
    if training:
        ds = ds.shuffle(min(len(df), 2048), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(tf_loader, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df, training=False)

# %% [markdown]
# ## U-Net model

# %%


def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_unet(input_shape=(256, 256, 1), base_filters=32):
    inputs = tf.keras.Input(input_shape)

    c1 = conv_block(inputs, base_filters)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, base_filters * 8)
    p4 = tf.keras.layers.MaxPooling2D()(c4)

    bn = conv_block(p4, base_filters * 16)
    bn = tf.keras.layers.Dropout(0.25)(bn)

    u4 = tf.keras.layers.UpSampling2D()(bn)
    u4 = tf.keras.layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, base_filters * 8)

    u3 = tf.keras.layers.UpSampling2D()(c5)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, base_filters * 4)

    u2 = tf.keras.layers.UpSampling2D()(c6)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, base_filters * 2)

    u1 = tf.keras.layers.UpSampling2D()(c7)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, base_filters)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c8)
    return tf.keras.Model(inputs, outputs, name="lung_tumor_unet_2d")


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * intersection + smooth) / (denominator + smooth))


def dice_bce_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + (1.0 - dice_coef(y_true, y_pred))


model = build_unet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), base_filters=BASE_FILTERS)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=dice_bce_loss,
    metrics=[dice_coef, tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="tumor_iou")],
)
model.summary()

# %% [markdown]
# ## Train

# %%
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef",
        mode="max",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(HISTORY_PATH),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# %% [markdown]
# ## Export model and preprocessing config

# %%
custom_objects = {"dice_coef": dice_coef, "dice_bce_loss": dice_bce_loss}
best_model = tf.keras.models.load_model(BEST_MODEL_PATH, custom_objects=custom_objects, compile=False)
try:
    best_model.save(H5_MODEL_PATH, include_optimizer=False)
except TypeError:
    best_model.save(H5_MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = []
tflite_model = converter.convert()
TFLITE_MODEL_PATH.write_bytes(tflite_model)

preprocess_config = {
    "task": "lung_tumor_segmentation_2d",
    "source_dataset": "rasoulisaeid/lung-cancer-segment / MSD Task06_Lung",
    "input_size": list(IMG_SIZE),
    "input_channels": 1,
    "hu_min": HU_MIN,
    "hu_max": HU_MAX,
    "normalization": "clip HU to [hu_min, hu_max], scale to [0, 1]",
    "output": "single-channel sigmoid tumor mask",
    "threshold": 0.5,
}
CONFIG_PATH.write_text(json.dumps(preprocess_config, indent=2), encoding="utf-8")

print("Saved:", BEST_MODEL_PATH)
print("Saved:", H5_MODEL_PATH)
print("Saved:", TFLITE_MODEL_PATH)
print("Saved:", CONFIG_PATH)

# %% [markdown]
# ## Visual check

# %%


def overlay_mask(image, mask, alpha=0.45):
    base = np.repeat(image, 3, axis=-1)
    overlay = base.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], mask[..., 0])
    overlay[..., 1] = overlay[..., 1] * (1 - alpha * mask[..., 0])
    overlay[..., 2] = overlay[..., 2] * (1 - alpha * mask[..., 0])
    return np.clip(overlay, 0, 1)


samples = list(val_ds.unbatch().take(6))
fig, axes = plt.subplots(len(samples), 3, figsize=(10, 3 * len(samples)))
if len(samples) == 1:
    axes = np.expand_dims(axes, axis=0)

for row, (image, mask) in enumerate(samples):
    pred = best_model.predict(image[None, ...], verbose=0)[0]
    pred_mask = (pred >= 0.5).astype(np.float32)

    axes[row, 0].imshow(image[..., 0], cmap="gray")
    axes[row, 0].set_title("CT slice")
    axes[row, 1].imshow(overlay_mask(image.numpy(), mask.numpy()))
    axes[row, 1].set_title("Ground truth")
    axes[row, 2].imshow(overlay_mask(image.numpy(), pred_mask))
    axes[row, 2].set_title("Prediction")

    for col in range(3):
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(PREDICTIONS_PNG, dpi=180)
plt.show()

print("Saved:", PREDICTIONS_PNG)

# %% [markdown]
# ## Download from Kaggle output
#
# After the run completes, open the notebook's right-side **Output** panel and download:
# - `lung_tumor_unet_best.keras`
# - `lung_tumor_unet_best.h5`
# - `lung_tumor_unet_float32.tflite`
# - `preprocess_config.json`
# - `sample_predictions.png`
