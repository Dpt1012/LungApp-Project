# LungCare Pi Camera

Realtime lung CT screening prototype for Raspberry Pi 4 using an OV camera connected directly to the Pi camera port. The project uses a LungCare PyTorch segmentation model to analyze a live camera feed and draw suspected regions on the displayed CT slice.

> This is an engineering prototype only. It is not a medical device and must not be used for diagnosis, treatment, or clinical decisions.

## What This Project Does

- Reads live frames from a Raspberry Pi OV camera through `rpicam/libcamera`.
- Runs a LungCare FPN EfficientNet-B4 segmentation model on CT slice images.
- Draws segmentation overlays and bounding boxes for suspected tumor regions.
- Supports ROI crop so the model can focus only on the CT slice area.
- Supports display scaling so small ROI regions are easier to view.
- Uses asynchronous inference so the camera preview does not freeze while the model is running.

## Hardware

- Raspberry Pi 4
- OV camera module connected directly to the Raspberry Pi camera connector
- Display or remote desktop session capable of showing OpenCV windows

The project does not use CanMV/K230 and does not use YOLO.

## Main Files

```text
lungcare_pi_camera_segment.py        # Main realtime Pi camera scanner
run_lungcare_pi_camera.sh            # Optional run helper
lungcarecamera/                      # Backup/collected LungCare files
```

Expected model file on the Pi:

```text
/home/pi/test/fpn_efficientnet_b4_gpu_p100_optimized_best.pt
```

## Raspberry Pi Setup

Install camera, OpenCV, Pillow, NumPy, and PyTorch packages:

```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-libcamera python3-opencv python3-pil python3-numpy python3-torch python3-torchvision rpicam-apps
```

Create a virtual environment that can use the system camera and PyTorch packages:

```bash
cd ~/test
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

Install the segmentation dependencies without reinstalling a large Torch wheel:

```bash
pip install --no-deps segmentation-models-pytorch timm efficientnet-pytorch pretrainedmodels
pip install --no-deps huggingface-hub safetensors tqdm packaging pyyaml filelock requests httpx httpcore h11 anyio sniffio certifi idna charset-normalizer urllib3
```

Check the environment:

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import segmentation_models_pytorch as smp; print('smp OK')"
```

## Copy Files To The Pi

Put these files in `/home/pi/test/`:

```text
lungcare_pi_camera_segment.py
fpn_efficientnet_b4_gpu_p100_optimized_best.pt
```

## Test The Camera

Check that the OV camera is detected:

```bash
rpicam-still -n -o test_cam.jpg
```

If the command prints `Still capture image received`, the camera is working.

## Run Realtime LungCare Scanner

Recommended command:

```bash
cd ~/test
source venv/bin/activate

rpicam-vid -t 0 --codec yuv420 --width 320 --height 240 --framerate 8 -n -o - | python3 lungcare_pi_camera_segment.py --realtime --camera-backend stdin-yuv --width 320 --height 240 --input-fps 8 --yuv-format gray --every-n-frames 40 --threshold 0.005 --box-min-area 4 --candidate-boxes --roi 105,45,135,150 --display-scale 1  --no-crop --torch-threads 1
```

The command uses the Y channel from the YUV camera stream. This gives a stable grayscale image, which is appropriate because CT slices and the LungCare model are grayscale-based.

## Important Runtime Options

| Option | Purpose |
| --- | --- |
| `--camera-backend stdin-yuv` | Reads frames piped from `rpicam-vid`. |
| `--yuv-format gray` | Uses only the Y/luminance channel from YUV420. |
| `--roi x,y,w,h` | Crops the camera frame before inference. |
| `--display-scale 2.5` | Enlarges the cropped preview window. |
| `--threshold 0.005` | Lowers the segmentation threshold for camera-on-screen tests. |
| `--candidate-boxes` | Draws heuristic candidate boxes when the model mask is empty. |
| `--box-min-area 4` | Minimum connected area for drawing a candidate box. |
| `--every-n-frames 40` | Runs the model every N frames. Larger values keep preview smoother. |
| `--torch-threads 1` | Limits CPU threads on Pi 4 to reduce UI lag. |

## Adjust The ROI

The ROI must contain only the CT slice area. If the program sees window borders, text, toolbars, or background, detection quality drops.

ROI format:

```text
--roi x,y,w,h
```

Example:

```bash
--roi 105,45,135,150
```

If the preview is too small after cropping, increase:

```bash
--display-scale 1
```

## Test A Single Image

```bash
python3 lungcare_pi_camera_segment.py --image test_cam.jpg --threshold 0.005 --candidate-boxes --box-min-area 4 --display-scale 2
```

Outputs are written to:

```text
/home/pi/test/lungcare_outputs/
```

## Notes And Limitations

- The model works best on clean CT slice images.
- A camera pointed at a monitor is a domain shift from the original CT training data.
- Reflections, monitor brightness, zoom level, window borders, and camera angle can reduce detection quality.
- On Raspberry Pi 4 CPU, the LungCare model can take several seconds per inference.
- Candidate boxes are only visual hints and can produce false positives.

## Safety Notice

This project is for learning, prototyping, and engineering demonstration. It does not provide medical advice. Any suspected abnormality must be reviewed by qualified medical professionals using validated clinical tools.

