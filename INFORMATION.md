# AI LungCare

Ung dung Flet ho tro:

- Chat voi Bac si AI LungCare qua Groq, linh hoat tieng Viet/tieng Anh.
- Tam soat nguy co bang Random Forest tu trieu chung.
- Segment u phoi tu anh/CT slice bang model `FPN EfficientNet-B4` da train.
- Hien `Image | Predicted Mask` va dien giai chan doan AI ho tro. Ket luan that van can bac si xac nhan.

## Copy sang may khac

Dung goi source portable:

```text
Lung_Cancer_Project_Portable_Source.zip
```

Giai nen zip ra mot thu muc bat ky. Thu muc phai co cac file chinh:

```text
main.py
lung_tumor_segmentation.py
rf_model.pkl
fpn_efficientnet_b4_gpu_p100_optimized_best.pt
requirements-*.txt
setup_windows.bat / run_windows.bat
setup_pi.sh / run_pi.sh
test_inputs_for_app
```

Khong copy thu muc `venv` tu may cu sang may moi. Moi may tu tao `.venv` rieng bang script setup.

## Chay tren Windows PC

Can Python 3.10+ ban 64-bit va internet lan dau de tai package.

1. Giai nen `Lung_Cancer_Project_Portable_Source.zip`.
2. Mo thu muc vua giai nen.
3. Chay:

```bat
setup_windows.bat
```

4. Sau khi setup xong, chay app:

```bat
run_windows.bat
```

Neu muon sua Groq key truoc khi mo app, sua file `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=openai/gpt-oss-120b
```

## Chay tren Raspberry Pi

Khuyen nghi Raspberry Pi OS 64-bit, Python 3.10+, RAM 4GB tro len. Segment bang CPU tren Pi se cham hon PC.

1. Giai nen `Lung_Cancer_Project_Portable_Source.zip` len Pi.
2. Mo terminal trong thu muc app.
3. Chay:

```bash
chmod +x setup_pi.sh run_pi.sh
./setup_pi.sh
./run_pi.sh
```

Mac dinh `run_pi.sh` chay Flet bang che do web tai:

```text
http://127.0.0.1:8550
```

Neu muon mo tu may tinh khac trong cung mang, chay tren Pi:

```bash
LUNGCARE_FLET_VIEW=server LUNGCARE_HOST=0.0.0.0 ./run_pi.sh
```

Sau do mo tren may tinh:

```text
http://<IP-cua-Pi>:8550
```

## Kiem tra moi truong

Sau khi setup, co the kiem tra nhanh:

```bash
python check_environment.py
```

Kiem tra ca viec load weight segment:

```bash
python check_environment.py --load-segmenter
```

## Anh test

Folder input-only de test app:

```text
test_inputs_for_app
```

Hay chon cac file `*_input.npy`, vi day la dung dinh dang/preprocessing cua notebook train FPN-B4.

Vi du:

```text
test_inputs_for_app/val58_slice287_input.npy
```

Sau khi segment, app se hien `Image | Predicted Mask`. Ground truth khong hien trong app.

## Dong goi lai goi portable

Neu da sua code/model va muon tao zip source portable moi:

```bash
python scripts/make_portable_zip.py
```

## Luu y y khoa

Ket qua segment, tam soat va chan doan AI chi co tinh chat ho tro. Ket luan y khoa cuoi cung phai do bac si va quy trinh chan doan lam sang thuc hien.
