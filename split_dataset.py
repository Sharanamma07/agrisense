"""
AgriSense — dataset split script.

Takes the raw PlantVillage download (one folder per class, all images
inside) and splits each class into train/ and val/ subfolders under
data/, in the layout train.py expects:

    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg

Usage:
    python split_dataset.py --source /path/to/raw_plantvillage --val_ratio 0.2

The raw PlantVillage download from Kaggle usually looks like:
    raw_plantvillage/
        Tomato___Late_blight/
            img1.jpg
            img2.jpg
        Tomato___healthy/
            img1.jpg
        ...
"""

import argparse
import os
import random
import shutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def split_dataset(source_dir, dest_dir, val_ratio, seed):
    random.seed(seed)

    class_names = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]
    if not class_names:
        raise ValueError(f"No class subfolders found in {source_dir}")

    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")

    print(f"Found {len(class_names)} classes. Splitting with val_ratio={val_ratio} ...")

    for cls in class_names:
        src_cls_dir = os.path.join(source_dir, cls)
        images = [
            f for f in os.listdir(src_cls_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ]
        random.shuffle(images)

        n_val = max(1, int(len(images) * val_ratio))
        val_images = images[:n_val]
        train_images = images[n_val:]

        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

        for f in train_images:
            shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(train_cls_dir, f))
        for f in val_images:
            shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(val_cls_dir, f))

        print(f"  {cls}: {len(train_images)} train, {len(val_images)} val")

    print(f"\nDone. Data written to {train_dir} and {val_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split raw PlantVillage dataset into train/val.")
    parser.add_argument("--source", required=True, help="Path to raw dataset (one folder per class)")
    parser.add_argument("--dest", default="data", help="Destination root (default: data/)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of each class for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    args = parser.parse_args()

    split_dataset(args.source, args.dest, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
