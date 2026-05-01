import os
import shutil
import random
import glob

CLEAN_DIR = "data/clean_vin"
TAMPERED_DIR = "data/tampered_vin"
OUTPUT_DIR = "data"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def make_dirs():
    for split in ["train", "val", "test"]:
        for label in ["Original", "Altered"]:
            os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)


def get_base_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    name = name.replace("_tampered_0", "")
    name = name.replace("_tampered_1", "")
    name = name.replace("_tampered_2", "")
    return name


def copy_file(src, dst_folder):
    if os.path.exists(src):
        shutil.copy2(src, dst_folder)


def split_dataset():
    make_dirs()

    clean_images = (
        glob.glob(os.path.join(CLEAN_DIR, "*.jpg")) +
        glob.glob(os.path.join(CLEAN_DIR, "*.jpeg")) +
        glob.glob(os.path.join(CLEAN_DIR, "*.png"))
    )

    random.shuffle(clean_images)

    total = len(clean_images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    split_map = {
        "train": clean_images[:train_end],
        "val": clean_images[train_end:val_end],
        "test": clean_images[val_end:]
    }

    for split, images in split_map.items():
        for clean_path in images:
            base_name = get_base_name(clean_path)

            copy_file(clean_path, os.path.join(OUTPUT_DIR, split, "Original"))

            tampered_matches = glob.glob(
                os.path.join(TAMPERED_DIR, f"{base_name}_tampered_*.*")
            )

            for tampered_path in tampered_matches:
                copy_file(tampered_path, os.path.join(OUTPUT_DIR, split, "Altered"))

    print("Dataset split completed successfully.")
    print(f"Total clean images: {total}")
    print(f"Train: {len(split_map['train'])}")
    print(f"Validation: {len(split_map['val'])}")
    print(f"Test: {len(split_map['test'])}")


if __name__ == "__main__":
    split_dataset()