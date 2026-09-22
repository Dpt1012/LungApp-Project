import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lung_tumor_segmentation import (
    LungTumorFpnB4,
    _display_uint8,
    _save_image_prediction_pair,
    _save_mask_png,
    _save_overlay,
)


DATASET_ZIP = Path(r"C:\Users\toanp\Downloads\archive.zip")
WEIGHTS = ROOT / "fpn_efficientnet_b4_gpu_p100_optimized_best.pt"
OUTPUT_DIR = ROOT / "fpn_b4_ground_truth_test_samples"


def parse_case_slice(path):
    parts = path.split("/")
    return parts[1], Path(path).stem


def panel_label(draw, text, x, y):
    draw.text((x + 8, y + 6), text, fill=(20, 20, 20))


def make_comparison(image_np, gt_mask, pred_mask, output_path, title):
    base = Image.fromarray(_display_uint8(image_np)).convert("RGB")
    gt_img = Image.fromarray((gt_mask > 0).astype(np.uint8) * 255).convert("RGB")
    pred_img = Image.fromarray((pred_mask > 0).astype(np.uint8) * 255).convert("RGB")

    width, height = base.size
    title_height = 34
    canvas = Image.new("RGB", (width * 3, height + title_height), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    panel_label(draw, f"Image {title}", 0, 0)
    panel_label(draw, "Ground Truth", width, 0)
    panel_label(draw, "Predicted Mask", width * 2, 0)
    canvas.paste(base, (0, title_height))
    canvas.paste(gt_img, (width, title_height))
    canvas.paste(pred_img, (width * 2, title_height))
    canvas.save(output_path)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in OUTPUT_DIR.glob("*"):
        if path.is_file():
            path.unlink()

    with zipfile.ZipFile(DATASET_ZIP) as dataset:
        mask_names = [
            name
            for name in dataset.namelist()
            if name.startswith("val/") and "/masks/" in name and name.endswith(".npy")
        ]

        mask_rows = []
        for mask_name in mask_names:
            mask = np.load(io.BytesIO(dataset.read(mask_name)), allow_pickle=False).astype(np.float32)
            pixels = int((mask > 0.5).sum())
            if pixels:
                mask_rows.append((pixels, mask_name))

        mask_rows.sort(reverse=True)
        val57_rows = [(pixels, name) for pixels, name in mask_rows if name.startswith("val/57/")]

        candidate_names = []
        for _pixels, name in mask_rows[:80] + val57_rows[:8]:
            if name not in candidate_names:
                candidate_names.append(name)

        images = []
        masks = []
        metas = []
        for mask_name in candidate_names:
            image_name = mask_name.replace("/masks/", "/data/")
            image_np = np.load(io.BytesIO(dataset.read(image_name)), allow_pickle=False).astype(np.float32)
            mask_np = np.load(io.BytesIO(dataset.read(mask_name)), allow_pickle=False).astype(np.float32)

            if image_np.ndim == 3:
                image_np = np.squeeze(image_np)
            if mask_np.ndim == 3:
                mask_np = np.squeeze(mask_np)

            images.append(image_np)
            masks.append((mask_np > 0.5).astype(np.uint8))
            metas.append((image_name, mask_name))

    segmenter = LungTumorFpnB4(str(WEIGHTS), batch_size=4)
    predictions, probabilities = segmenter.predict_slices(images, "npy")

    stats = []
    for image_np, gt_mask, pred_mask, prob, meta in zip(images, masks, predictions, probabilities, metas):
        pred_mask = (pred_mask > 0).astype(np.uint8)
        intersection = int(((pred_mask > 0) & (gt_mask > 0)).sum())
        pred_pixels = int((pred_mask > 0).sum())
        gt_pixels = int((gt_mask > 0).sum())
        union = int(((pred_mask > 0) | (gt_mask > 0)).sum())
        dice = (2 * intersection / (pred_pixels + gt_pixels)) if (pred_pixels + gt_pixels) else 0.0
        iou = (intersection / union) if union else 0.0
        stats.append(
            {
                "image_name": meta[0],
                "mask_name": meta[1],
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "dice": dice,
                "iou": iou,
                "prob_max": float(prob),
                "image": image_np,
                "gt": gt_mask,
                "pred": pred_mask,
            }
        )

    selected = [
        sample
        for sample in sorted(stats, key=lambda item: (item["dice"], item["gt_pixels"], item["prob_max"]), reverse=True)
        if sample["pred_pixels"] > 0
    ][:8]
    if len(selected) < 8:
        extras = [
            sample
            for sample in sorted(stats, key=lambda item: (item["gt_pixels"], item["prob_max"]), reverse=True)
            if sample not in selected
        ]
        selected.extend(extras[: 8 - len(selected)])

    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample", "source_image", "gt_pixels", "pred_pixels", "dice", "iou", "prob_max"],
        )
        writer.writeheader()

        readme_lines = [
            "# FPN-B4 Ground Truth Test Samples",
            "",
            "Dung cac file `*_input.npy` de test app vi day la dung dinh dang/preprocessing cua notebook.",
            "Cac file PNG dung de xem nhanh. `*_gt_mask.*` la ground truth, khong phai input app.",
            "",
            "Bang mau da chon:",
        ]

        for sample in selected:
            case_id, slice_id = parse_case_slice(sample["image_name"])
            sample_name = f"val{case_id}_slice{int(slice_id):03d}"

            np.save(OUTPUT_DIR / f"{sample_name}_input.npy", sample["image"])
            np.save(OUTPUT_DIR / f"{sample_name}_gt_mask.npy", sample["gt"].astype(np.uint8))
            Image.fromarray(_display_uint8(sample["image"])).save(OUTPUT_DIR / f"{sample_name}_input.png")
            _save_mask_png(sample["gt"], OUTPUT_DIR / f"{sample_name}_gt_mask.png")
            _save_mask_png(sample["pred"], OUTPUT_DIR / f"{sample_name}_pred_mask.png")
            _save_overlay(sample["image"], sample["gt"], OUTPUT_DIR / f"{sample_name}_gt_overlay.png")
            _save_overlay(sample["image"], sample["pred"], OUTPUT_DIR / f"{sample_name}_pred_overlay.png")
            _save_image_prediction_pair(
                sample["image"],
                sample["pred"],
                OUTPUT_DIR / f"{sample_name}_input_vs_predicted_mask.png",
            )
            make_comparison(
                sample["image"],
                sample["gt"],
                sample["pred"],
                OUTPUT_DIR / f"{sample_name}_comparison.png",
                f"val/{case_id}/{slice_id}.npy",
            )

            writer.writerow(
                {
                    "sample": sample_name,
                    "source_image": sample["image_name"],
                    "gt_pixels": sample["gt_pixels"],
                    "pred_pixels": sample["pred_pixels"],
                    "dice": round(sample["dice"], 4),
                    "iou": round(sample["iou"], 4),
                    "prob_max": round(sample["prob_max"], 4),
                }
            )
            readme_lines.append(
                f"- `{sample_name}_input.npy`: GT pixels={sample['gt_pixels']}, "
                f"pred pixels={sample['pred_pixels']}, Dice={sample['dice']:.4f}, "
                f"prob max={sample['prob_max']:.4f}. App se hien kieu "
                f"`{sample_name}_input_vs_predicted_mask.png`; file `{sample_name}_comparison.png` "
                "chi de doi chieu voi ground truth."
            )

    (OUTPUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"Created {OUTPUT_DIR}")
    for sample in selected:
        case_id, slice_id = parse_case_slice(sample["image_name"])
        print(
            f"val/{case_id}/{slice_id}.npy "
            f"gt={sample['gt_pixels']} pred={sample['pred_pixels']} "
            f"dice={sample['dice']:.4f} prob={sample['prob_max']:.4f}"
        )


if __name__ == "__main__":
    main()
