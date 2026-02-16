"""Download and organize polyp segmentation datasets.

Datasets are downloaded from Google Drive (standard sources used in
PraNet, Polyp-PVT, and other polyp segmentation papers).

Expected output structure:
    datasets/
    ├── Kvasir/
    │   ├── images/
    │   └── masks/
    ├── CVC-ClinicDB/
    │   ├── images/
    │   └── masks/
    ├── CVC-ColonDB/
    │   ├── images/
    │   └── masks/
    ├── ETIS-LaribPolypDB/
    │   ├── images/
    │   └── masks/
    └── CVC-300/
        ├── images/
        └── masks/
"""

import os
import shutil
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None


# Google Drive file IDs for polyp datasets (from PraNet repository)
DATASET_URLS = {
    "TrainDataset": "https://drive.google.com/uc?id=1lODorfB33jbd-im-qrtUgWnZXxB94F55",
    "TestDataset": "https://drive.google.com/uc?id=1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R",
}


def download_datasets(output_dir="./datasets"):
    """Download and extract polyp segmentation datasets.

    Args:
        output_dir: directory to save datasets
    """
    if gdown is None:
        print("Please install gdown: pip install gdown")
        print("Then re-run this script.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, url in DATASET_URLS.items():
        zip_path = output_dir / f"{name}.zip"

        if not zip_path.exists():
            print(f"Downloading {name}...")
            gdown.download(url, str(zip_path), quiet=False)
        else:
            print(f"{name}.zip already exists, skipping download.")

        # Extract
        extract_dir = output_dir / name
        if not extract_dir.exists():
            print(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(output_dir))
            print(f"Extracted to {extract_dir}")
        else:
            print(f"{extract_dir} already exists, skipping extraction.")

    # Reorganize into per-center structure
    _reorganize_datasets(output_dir)

    print("\nDataset setup complete!")
    _print_stats(output_dir)


def _reorganize_datasets(output_dir):
    """Reorganize the downloaded data into per-center folders.

    The downloaded TrainDataset has all training images pooled together.
    TestDataset already has per-center folders. We need to create
    per-center training folders for meta-learning.

    For meta-learning, we use the TestDataset's per-center structure
    as the primary data source since it's already organized by center.
    """
    output_dir = Path(output_dir)
    test_dir = output_dir / "TestDataset"

    if not test_dir.exists():
        print("TestDataset not found. Please download first.")
        return

    # Copy test dataset centers to top-level for easy access
    centers = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB",
               "ETIS-LaribPolypDB", "Kvasir"]

    for center in centers:
        src = test_dir / center
        dst = output_dir / center

        if src.exists() and not dst.exists():
            print(f"Organizing {center}...")
            shutil.copytree(str(src), str(dst))

    # Handle TrainDataset: this contains pooled Kvasir + CVC-ClinicDB images
    # For meta-learning, we already have per-center data from TestDataset
    # The TrainDataset images can supplement if needed
    train_dir = output_dir / "TrainDataset"
    if train_dir.exists():
        print("\nNote: TrainDataset contains pooled training images.")
        print("For meta-learning, using per-center data from TestDataset.")
        print("TrainDataset is kept for baseline (non-meta) training.")


def _print_stats(output_dir):
    """Print dataset statistics."""
    output_dir = Path(output_dir)
    centers = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB",
               "ETIS-LaribPolypDB", "CVC-300"]

    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    for center in centers:
        img_dir = output_dir / center / "images"
        if img_dir.exists():
            n_images = len(list(img_dir.iterdir()))
            print(f"  {center:25s}: {n_images:4d} images")
        else:
            print(f"  {center:25s}: NOT FOUND")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download polyp datasets")
    parser.add_argument(
        "--output-dir", type=str, default="./datasets",
        help="Output directory for datasets"
    )
    args = parser.parse_args()

    download_datasets(args.output_dir)
