# Train segmentation model tren Kaggle

## Chuan bi tren Kaggle

1. Tao Notebook moi tren Kaggle.
2. Vao **Add Input** va them dataset: `rasoulisaeid/lung-cancer-segment`.
3. Vao **Notebook options**:
   - Accelerator: GPU T4 hoac P100
   - Internet: On neu Kaggle image chua co `nibabel`
4. Upload hoac copy noi dung file `kaggle_lung_tumor_segmentation_unet.py` vao notebook.
5. Chay all cells.

Neu can debug layout dataset truoc khi train, copy tung cell trong:

```text
KAGGLE_NOTEBOOK_CELLS.md
```

Screenshot cua ban co layout `train/0/data` va `train/0/masks`; ban script moi da ho tro layout nay.

## Output can tai ve

Sau khi train xong, tai cac file trong `/kaggle/working`:

- `lung_tumor_unet_best.keras`
- `lung_tumor_unet_best.h5`
- `lung_tumor_unet_float32.tflite`
- `preprocess_config.json`
- `sample_predictions.png`
- `training_history.csv`

Nen copy cac file model ve project local vao thu muc:

```text
models/segmentation/
```

## Luu y quan trong

Dataset nay la CT 3D co mask u phoi, khac voi model classifier hien tai dang dung anh 2D grayscale. Model train tu notebook nay phu hop de segment tumor tren slice CT da tien xu ly theo:

- clip HU tu `-1000` den `400`
- resize ve `256x256`
- output la mask sigmoid 1 kenh

Khong dung model nay nhu chan doan y khoa. Day chi la cong cu ho tro tam soat/nghien cuu.
