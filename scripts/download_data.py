"""Download public, CC BY 4.0 dataset 529 from the UCI Machine Learning Repository.

Run from the repository root: python scripts/download_data.py
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

UCI_ZIP = (
    "https://archive.ics.uci.edu/static/public/529/"
    "early+stage+diabetes+risk+prediction+dataset.zip"
)
DESTINATION = Path("data/raw/diabetes_data_upload.csv")


def main() -> None:
    if DESTINATION.exists():
        print(f"Already downloaded: {DESTINATION}")
        return
    print("Downloading dataset from the UCI Machine Learning Repository...")
    try:
        request = Request(UCI_ZIP, headers={"User-Agent": "Mozilla/5.0 (research/1.0)"})
        with urlopen(request, timeout=30) as response:
            zip_bytes = response.read()
        with ZipFile(BytesIO(zip_bytes)) as archive:
            matches = [path for path in archive.namelist()
                       if Path(path).name.lower() == "diabetes_data_upload.csv"]
            if not matches:
                raise FileNotFoundError("CSV not found inside downloaded UCI ZIP archive.")
            content = archive.read(matches[0])
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        DESTINATION.write_bytes(content)
        print(f"Saved {DESTINATION} ({len(content)} bytes)")
    except Exception:
        print("Automatic download failed. Manually download 'diabetes_data_upload.csv' ")
        print("from https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset")
        print(f"and put it at {DESTINATION}, then rerun training.")
        raise


if __name__ == "__main__":
    main()
