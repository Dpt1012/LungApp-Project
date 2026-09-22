# Lung Tumor Segmentation Test Samples

Dung file `case0_tumor_subvolume.nii.gz` trong app truoc. Day la mot CT subvolume ngan, cat quanh cac slice co u tu dataset Lung Tumor Segmentation.

Files:
- `case0_tumor_subvolume.nii.gz`: input nen dung de test app.
- `case0_tumor_subvolume.npy`: cung volume duoi dang NumPy.
- `case0_tumor_subvolume_gt_mask.*`: mask ground-truth de doi chieu, khong phai input app.
- `case0_slice*_input.png`: anh preview 2D cua cac slice CT.
- `case0_slice*_gt_mask.png`: mask ground-truth cua tung slice.
- `case0_slice*_gt_overlay.png`: overlay anh + ground-truth mask de nhin nhanh vung u.

Luu y: day la mau CT u phoi, khong phai u nao. Checkpoint SegNet trong app duoc train cho lung tumor segmentation, nen anh u nao se khong phu hop voi model nay.
