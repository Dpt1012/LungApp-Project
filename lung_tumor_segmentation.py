import os
import tempfile
from pathlib import Path

import numpy as np


FPN_INPUT_SIZE = (256, 256)
FPN_THRESHOLD = 0.5


def _sanitize_name(name):
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(keep).strip("._") or "scan"


def _result_dir():
    path = Path(tempfile.gettempdir()) / "lung_tumor_segmentation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _display_uint8(slice_array):
    arr = np.asarray(slice_array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(arr, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(arr))
        high = float(np.max(arr))
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _save_overlay(slice_array, mask, output_path):
    from PIL import Image

    image = Image.fromarray(_display_uint8(slice_array)).convert("RGBA")
    mask_img = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 180, mode="L")
    if mask_img.size != image.size:
        resampling = getattr(Image, "Resampling", Image).NEAREST
        mask_img = mask_img.resize(image.size, resampling)

    red = Image.new("RGBA", image.size, (235, 52, 78, 0))
    red.putalpha(mask_img)
    overlay = Image.alpha_composite(image, red)
    overlay.save(output_path)


def _save_mask_png(mask, output_path):
    from PIL import Image

    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(output_path)


def _save_image_prediction_pair(slice_array, mask, output_path):
    from PIL import Image, ImageDraw

    image = Image.fromarray(_display_uint8(slice_array)).convert("RGB")
    mask_image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).convert("RGB")
    if mask_image.size != image.size:
        resampling = getattr(Image, "Resampling", Image).NEAREST
        mask_image = mask_image.resize(image.size, resampling)

    width, height = image.size
    title_height = 34
    canvas = Image.new("RGB", (width * 2, height + title_height), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "Image", fill=(20, 20, 20))
    draw.text((width + 8, 7), "Predicted Mask", fill=(20, 20, 20))
    canvas.paste(image, (0, title_height))
    canvas.paste(mask_image, (width, title_height))
    canvas.save(output_path)


def _resize_float(array, size):
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = Image.fromarray(np.asarray(array, dtype=np.float32))
    return np.asarray(image.resize(size, resampling), dtype=np.float32)


def _resize_mask(mask, shape):
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).NEAREST
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    return (np.asarray(image.resize((shape[1], shape[0]), resampling)) > 0).astype(np.uint8)


class LungTumorFpnB4:
    def __init__(self, weights_path, batch_size=4):
        import torch
        import segmentation_models_pytorch as smp

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.encoder_name = checkpoint.get("encoder_name", "efficientnet-b4")
            self.image_size = int(checkpoint.get("image_size", FPN_INPUT_SIZE[0]))
            self.best_val_dice = checkpoint.get("best_val_dice")
        else:
            state_dict = checkpoint
            self.encoder_name = "efficientnet-b4"
            self.image_size = FPN_INPUT_SIZE[0]
            self.best_val_dice = None

        self.model = smp.FPN(
            self.encoder_name,
            encoder_weights=None,
            in_channels=1,
            classes=1,
        ).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @staticmethod
    def _scale_slice(slice_array, input_kind):
        arr = np.asarray(slice_array, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 3:
            arr = np.squeeze(arr)

        if arr.size == 0:
            return arr

        min_value = float(np.nanmin(arr))
        max_value = float(np.nanmax(arr))
        if min_value >= 0.0 and max_value <= 1.5:
            return np.clip(arr, 0.0, 1.0)
        if min_value >= 0.0 and max_value <= 255.0:
            return np.clip(arr / 255.0, 0.0, 1.0)

        # The trained notebook used preprocessed [0,1] NPY slices. For raw CT/NIfTI
        # or other ranges, use a display-style robust normalization as a best-effort fallback.
        low, high = np.percentile(arr, [1, 99])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = min_value
            high = max_value
        if high <= low:
            return np.zeros(arr.shape, dtype=np.float32)
        return np.clip((arr - low) / (high - low), 0.0, 1.0).astype(np.float32)

    def _prepare_slice(self, slice_array, input_kind):
        resized = _resize_float(self._scale_slice(slice_array, input_kind), (self.image_size, self.image_size))
        normalized = (resized - 0.5) / 0.5
        return normalized[np.newaxis, :, :].astype(np.float32, copy=False)

    def _predict_prepared_batch(self, prepared_batch):
        torch = self.torch
        tensor = torch.from_numpy(np.stack(prepared_batch, axis=0)).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits.float())[:, 0]
            masks = (probabilities >= FPN_THRESHOLD).to(torch.uint8)
        return masks.cpu().numpy().astype(np.uint8), probabilities.cpu().numpy().astype(np.float32)

    def predict_slices(self, slices, input_kind):
        masks = []
        probabilities = []
        for start in range(0, len(slices), self.batch_size):
            current = slices[start:start + self.batch_size]
            prepared = [self._prepare_slice(slice_array, input_kind) for slice_array in current]
            batch_masks, batch_probs = self._predict_prepared_batch(prepared)
            for slice_array, mask, prob in zip(current, batch_masks, batch_probs):
                masks.append(_resize_mask(mask, slice_array.shape[:2]))
                probabilities.append(float(np.nanmax(prob)) if prob.size else 0.0)
        return masks, probabilities

    def segment_volume(self, volume, input_kind):
        data = np.asarray(volume, dtype=np.float32)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        if data.ndim == 3 and data.shape[-1] in (3, 4) and input_kind in ("image", "npy"):
            data = np.mean(data[..., :3], axis=-1)[:, :, np.newaxis]
        if data.ndim != 3:
            raise ValueError(f"Expected a 2D slice or 3D volume, got shape {data.shape}.")

        mask_volume = np.zeros(data.shape, dtype=np.uint8)
        max_probability = 0.0
        for start in range(0, data.shape[-1], self.batch_size):
            indices = list(range(start, min(start + self.batch_size, data.shape[-1])))
            slices = [data[:, :, idx] for idx in indices]
            masks, probabilities = self.predict_slices(slices, input_kind)
            for idx, mask, probability in zip(indices, masks, probabilities):
                mask_volume[:, :, idx] = mask
                max_probability = max(max_probability, probability)

        areas = mask_volume.reshape(-1, mask_volume.shape[-1]).sum(axis=0)
        preview_idx = int(np.argmax(areas)) if int(areas.max(initial=0)) > 0 else data.shape[-1] // 2
        return mask_volume, preview_idx, max_probability


_SEGMENTER_CACHE = {}


def get_segmenter(weights_path):
    weights_path = os.path.abspath(weights_path)
    if weights_path not in _SEGMENTER_CACHE:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Segmentation model not found: {weights_path}")
        _SEGMENTER_CACHE[weights_path] = LungTumorFpnB4(weights_path)
    return _SEGMENTER_CACHE[weights_path]


def _strip_nii_suffix(path):
    name = Path(path).name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def segment_lung_tumor_file(file_path, weights_path):
    from PIL import Image, ImageOps

    file_path = os.path.abspath(file_path)
    lower_name = os.path.basename(file_path).lower()
    safe_stem = _sanitize_name(_strip_nii_suffix(file_path))
    outputs = _result_dir()
    segmenter = get_segmenter(weights_path)

    if lower_name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        image = nib.load(file_path)
        data = image.get_fdata(dtype=np.float32)
        orientation = "".join(nib.aff2axcodes(image.affine))
        mask_volume, preview_idx, max_probability = segmenter.segment_volume(data, "nifti")

        header = image.header.copy()
        header.set_data_dtype(np.uint8)
        mask_path = outputs / f"{safe_stem}_tumor_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask_volume.astype(np.uint8), image.affine, header), mask_path)

        overlay_path = outputs / f"{safe_stem}_overlay_slice_{preview_idx}.png"
        prediction_mask_path = outputs / f"{safe_stem}_predicted_mask_slice_{preview_idx}.png"
        comparison_path = outputs / f"{safe_stem}_input_vs_predicted_mask_slice_{preview_idx}.png"
        _save_overlay(data[:, :, preview_idx], mask_volume[:, :, preview_idx], overlay_path)
        _save_mask_png(mask_volume[:, :, preview_idx], prediction_mask_path)
        _save_image_prediction_pair(data[:, :, preview_idx], mask_volume[:, :, preview_idx], comparison_path)
        kind = "nifti"

    elif lower_name.endswith(".npy"):
        data = np.load(file_path, allow_pickle=False).astype(np.float32)
        mask_volume, preview_idx, max_probability = segmenter.segment_volume(data, "npy")
        mask_path = outputs / f"{safe_stem}_tumor_mask.npy"
        np.save(mask_path, mask_volume.squeeze() if mask_volume.shape[-1] == 1 else mask_volume)

        preview_data = data if data.ndim == 2 else data[:, :, preview_idx]
        preview_mask = mask_volume[:, :, 0] if mask_volume.shape[-1] == 1 else mask_volume[:, :, preview_idx]
        overlay_path = outputs / f"{safe_stem}_overlay.png"
        prediction_mask_path = outputs / f"{safe_stem}_predicted_mask.png"
        comparison_path = outputs / f"{safe_stem}_input_vs_predicted_mask.png"
        _save_overlay(preview_data, preview_mask, overlay_path)
        _save_mask_png(preview_mask, prediction_mask_path)
        _save_image_prediction_pair(preview_data, preview_mask, comparison_path)
        kind = "npy"
        orientation = ""

    else:
        image = ImageOps.exif_transpose(Image.open(file_path)).convert("L")
        data = np.asarray(image, dtype=np.float32)
        mask_volume, preview_idx, max_probability = segmenter.segment_volume(data, "image")
        mask = mask_volume[:, :, 0]
        mask_path = outputs / f"{safe_stem}_tumor_mask.png"
        overlay_path = outputs / f"{safe_stem}_overlay.png"
        prediction_mask_path = mask_path
        comparison_path = outputs / f"{safe_stem}_input_vs_predicted_mask.png"
        _save_mask_png(mask, mask_path)
        _save_overlay(data, mask, overlay_path)
        _save_image_prediction_pair(data, mask, comparison_path)
        kind = "image"
        orientation = ""

    tumor_voxels = int(mask_volume.sum())
    total_voxels = int(mask_volume.size)
    tumor_percent = (tumor_voxels / total_voxels * 100.0) if total_voxels else 0.0
    return {
        "kind": kind,
        "model": "FPN EfficientNet-B4",
        "overlay_path": str(overlay_path),
        "comparison_path": str(comparison_path),
        "prediction_mask_path": str(prediction_mask_path),
        "mask_path": str(mask_path),
        "preview_slice": int(preview_idx),
        "slices_processed": int(mask_volume.shape[-1]),
        "tumor_voxels": tumor_voxels,
        "tumor_percent": round(tumor_percent, 3),
        "max_probability": round(max_probability * 100.0, 1),
        "has_tumor": tumor_voxels > 0,
        "orientation": orientation,
        "best_val_dice": segmenter.best_val_dice,
    }
