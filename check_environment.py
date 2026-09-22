import argparse
import importlib.util
import platform
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "main.py",
    "lung_tumor_segmentation.py",
    "rf_model.pkl",
    "fpn_efficientnet_b4_gpu_p100_optimized_best.pt",
]

REQUIRED_IMPORTS = [
    ("flet", "flet"),
    ("flet_charts", "flet-charts"),
    ("plotly", "plotly"),
    ("dotenv", "python-dotenv"),
    ("openai", "openai"),
    ("sklearn", "scikit-learn"),
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("segmentation_models_pytorch", "segmentation-models-pytorch"),
    ("efficientnet_pytorch", "efficientnet-pytorch"),
    ("nibabel", "nibabel"),
]

OPTIONAL_IMPORTS = [
    ("flet_audio_recorder", "flet-audio-recorder"),
]


def has_module(module_name):
    return importlib.util.find_spec(module_name) is not None


def check_files():
    missing = []
    for rel_path in REQUIRED_FILES:
        path = APP_DIR / rel_path
        if path.exists():
            print(f"[OK] file: {rel_path}")
        else:
            print(f"[MISS] file: {rel_path}")
            missing.append(rel_path)
    if (APP_DIR / ".env").exists():
        print("[OK] file: .env")
    else:
        print("[WARN] file: .env not found. The app can still ask for the Groq API key.")
    return missing


def check_imports():
    missing = []
    for module_name, package_name in REQUIRED_IMPORTS:
        if has_module(module_name):
            print(f"[OK] package: {package_name}")
        else:
            print(f"[MISS] package: {package_name}")
            missing.append(package_name)
    for module_name, package_name in OPTIONAL_IMPORTS:
        if has_module(module_name):
            print(f"[OK] optional package: {package_name}")
        else:
            print(f"[WARN] optional package: {package_name} not installed. Voice input will be disabled.")
    return missing


def print_runtime_info():
    print(f"[INFO] python: {sys.version.split()[0]}")
    print(f"[INFO] platform: {platform.platform()}")
    print(f"[INFO] machine: {platform.machine()}")
    print(f"[INFO] architecture: {platform.architecture()[0]}")
    if sys.maxsize <= 2**32:
        print("[WARN] 32-bit Python detected. Use 64-bit OS/Python for the segmentation model.")
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] torch: {torch.__version__} ({device})")
    except Exception as ex:
        print(f"[WARN] could not read torch runtime: {ex}")


def load_segmenter():
    from lung_tumor_segmentation import get_segmenter

    weights = APP_DIR / "fpn_efficientnet_b4_gpu_p100_optimized_best.pt"
    get_segmenter(str(weights))
    print("[OK] segmentation model loaded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-segmenter", action="store_true", help="Also load the FPN-B4 weights.")
    args = parser.parse_args()

    print_runtime_info()
    missing_files = check_files()
    missing_packages = check_imports()

    if args.load_segmenter and not missing_files and not missing_packages:
        load_segmenter()

    if missing_files or missing_packages:
        print()
        print("[FAIL] Environment is incomplete.")
        return 1

    print()
    print("[OK] Environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
