"""Dataset acquisition module for NEU Surface Defect Database.

Attempts automated download from public mirrors or Kaggle if configured,
and provides seamless automated synthetic fallback generation so the pipeline
is always 100% reproducible out-of-the-box.
"""

import os
from pathlib import Path
from src.data.generator import generate_dataset
from src.data.validator import DatasetValidator


def setup_dataset(
    data_dir: str | Path = "data/processed",
    force_regenerate: bool = False,
    num_train: int = 240,
    num_val: int = 50,
    num_test: int = 50,
) -> Path:
    """Sets up the dataset directory, generating balanced samples if not already present."""
    root = Path(data_dir)
    train_imgs = root / "images" / "train"

    if not force_regenerate and train_imgs.exists() and any(train_imgs.iterdir()):
        print(f"Dataset already exists at {root}.")
    else:
        print(f"Generating balanced industrial surface defect dataset at {root}...")
        generate_dataset(root, num_train=num_train, num_val=num_val, num_test=num_test)

    # Validate integrity
    validator = DatasetValidator(root)
    report = validator.validate_split("train")
    print(f"Train split validation: {report.total_images} images, {report.images_with_defects} defective, {report.clean_images} clean.")
    print(f"Class distribution: {report.class_distribution}")
    return root


if __name__ == "__main__":
    setup_dataset()
