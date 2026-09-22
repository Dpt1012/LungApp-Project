import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "Lung_Cancer_Project_Portable_Source.zip"
BASE = "Lung_Cancer_Project_Portable"

FILES = [
    "main.py",
    "lung_tumor_segmentation.py",
    "rf_model.pkl",
    "fpn_efficientnet_b4_gpu_p100_optimized_best.pt",
    ".env.example",
    "requirements.txt",
    "requirements-common.txt",
    "requirements-windows.txt",
    "requirements-pi.txt",
    "setup_windows.bat",
    "run_windows.bat",
    "setup_pi.sh",
    "run_pi.sh",
    "check_environment.py",
    "README.md",
    "scripts/make_portable_zip.py",
]

FOLDERS = [
    "test_inputs_for_app",
]


def iter_package_files():
    for rel in FILES:
        path = ROOT / rel
        if path.exists():
            yield path, Path(BASE) / rel
    for folder in FOLDERS:
        root_folder = ROOT / folder
        if not root_folder.exists():
            continue
        for path in root_folder.rglob("*"):
            if path.is_file():
                yield path, Path(BASE) / folder / path.relative_to(root_folder)


def main():
    files = list(iter_package_files())
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print(f"Packaging {len(files)} files into {ZIP_PATH.name}...")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        for path, arcname in files:
            zf.write(path, arcname.as_posix())

    print(f"Created: {ZIP_PATH}")
    print(f"Size: {ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
