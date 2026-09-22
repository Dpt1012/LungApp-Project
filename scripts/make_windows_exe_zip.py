import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build" / "windows"
ZIP_PATH = ROOT / "Lung_Cancer_Project_Windows_EXE_Ready.zip"
BASE = "Lung_Cancer_Project_Windows"
TEST_INPUTS = ROOT / "test_inputs_for_app"

EXCLUDED_APP_NAMES = {
    ".env",
    "Lung_Cancer_Project_Windows.zip",
    "mri_cnn_model.h5",
    "lung_tumor_segnet_state.pt",
}
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    "mo_hinh_nha_3_tang",
}


def iter_windows_files():
    readme = ROOT / "README_WINDOWS_EXE.txt"
    if readme.exists():
        yield readme, Path(BASE) / "README_WINDOWS_EXE.txt"

    for path in SOURCE.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(SOURCE)
        parts = rel.parts
        if rel.as_posix() == "README_WINDOWS_EXE.txt":
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in parts):
            continue
        if len(parts) >= 2 and parts[0] == "app" and parts[-1] in EXCLUDED_APP_NAMES:
            continue
        yield path, Path(BASE) / rel

    if TEST_INPUTS.exists():
        for path in TEST_INPUTS.rglob("*"):
            if path.is_file():
                yield path, Path(BASE) / "test_inputs_for_app" / path.relative_to(TEST_INPUTS)


def main():
    files = list(iter_windows_files())
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    print(f"Packaging {len(files)} files into {ZIP_PATH.name}...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        for index, (path, arcname) in enumerate(files, 1):
            zf.write(path, arcname.as_posix())
            if index % 1000 == 0:
                print(f"  added {index}/{len(files)} files", flush=True)

    print(f"Created: {ZIP_PATH}", flush=True)
    print(f"Size: {ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
