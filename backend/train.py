import sys
import json
import zipfile
import random
import numpy as np
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    print("ERROR: tensorflow not installed. Run: pip install tensorflow==2.20.0")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install pillow")
    sys.exit(1)

# Configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 32
NUM_CLASSES = 10  # first 10 GTSRB classes
EPOCHS = 10
BATCH_SIZE = 32
DATASET_DIR = Path("../dataset")
SAMPLE_DIR = Path("../sample_images")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "traffic_sign_model.h5"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
ACCURACY_PATH = MODEL_DIR / "accuracy.txt"
METRICS_PATH = MODEL_DIR / "metrics.json"

CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing"
}

MODEL_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True, parents=True)
SAMPLE_DIR.mkdir(exist_ok=True, parents=True)


def download_dataset():
    """Download GTSRB training images (official server)"""
    zip_path = DATASET_DIR / "GTSRB_Final_Training_Images.zip"
    extract_path = DATASET_DIR / "GTSRB"

    if extract_path.exists() and any(extract_path.iterdir()):
        print("Dataset already extracted, skipping download.")
        return extract_path

    if not zip_path.exists():
        print("Downloading GTSRB dataset (this may take a while)...")
        import urllib.request
        url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
        try:
            urllib.request.urlretrieve(
                url, zip_path,
                reporthook=lambda b, bs, t: print(f"\r  Downloaded {b*bs/1e6:.1f} MB of {t/1e6:.1f} MB", end="")
            )
            print()
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please manually download the dataset to ../dataset/GTSRB_Final_Training_Images.zip")
            sys.exit(1)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATASET_DIR)

    return extract_path


def load_dataset(dataset_path):
    """Load images of the first NUM_CLASSES classes.
    Returns X, y, groups (track id per image), paths."""
    X, y, groups, paths = [], [], [], []

    gtsrb_path = dataset_path / "GTSRB" / "Final_Training" / "Images"
    if not gtsrb_path.exists():
        for p in dataset_path.rglob("Images"):
            if p.is_dir():
                gtsrb_path = p
                break

    print(f"Loading data from: {gtsrb_path}")

    for class_id in range(NUM_CLASSES):
        class_dir = gtsrb_path / f"{class_id:05d}"
        if not class_dir.exists():
            print(f"  Warning: Class dir {class_dir} not found, skipping")
            continue

        ppm_files = sorted(class_dir.glob("*.ppm"))
        print(f"  Class {class_id} ({CLASS_NAMES[class_id]}): {len(ppm_files)} images")

        for ppm_file in ppm_files:
            try:
                img = Image.open(ppm_file).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr)
                y.append(class_id)
                # GTSRB filename: <track>_<frame>.ppm -> all frames of one track are near-identical
                groups.append(f"{class_id}_{ppm_file.stem.split('_')[0]}")
                paths.append(str(ppm_file))
            except Exception:
                pass  # skip bad images

    return np.array(X), np.array(y), np.array(groups), paths


def split_by_track(y, groups, test_frac=0.2, val_frac=0.1):
    """Split so that no track (same physical sign) appears in more than one split."""
    rng = np.random.RandomState(SEED)
    split = np.array(["train"] * len(y), dtype=object)
    for c in np.unique(y):
        class_groups = np.unique(groups[y == c])
        rng.shuffle(class_groups)
        n_test = max(1, int(round(len(class_groups) * test_frac)))
        n_val = max(1, int(round(len(class_groups) * val_frac)))
        test_g = set(class_groups[:n_test])
        val_g = set(class_groups[n_test:n_test + n_val])
        for i in np.where(y == c)[0]:
            if groups[i] in test_g:
                split[i] = "test"
            elif groups[i] in val_g:
                split[i] = "val"
    return split


def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def save_sample_images(y, paths, test_idx):
    """Save up to 3 TEST images per class for honest demo uploads."""
    rng = np.random.RandomState(SEED)
    for c in range(NUM_CLASSES):
        cand = [i for i in test_idx if y[i] == c]
        rng.shuffle(cand)
        for n, i in enumerate(cand[:3], start=1):
            Image.open(paths[i]).convert("RGB").save(SAMPLE_DIR / f"class{c:02d}_example{n}.png")
    print(f"Sample test images saved to {SAMPLE_DIR.resolve()}")


def main():
    print("=" * 60)
    print("Traffic Sign Recognition - CNN Training")
    print("=" * 60)

    dataset_path = download_dataset()

    print("\nLoading dataset...")
    X, y, groups, paths = load_dataset(DATASET_DIR)

    if len(X) == 0:
        print("ERROR: No images loaded. Check dataset path.")
        sys.exit(1)

    print(f"\nDataset loaded: {len(X)} images, {NUM_CLASSES} classes")

    split = split_by_track(y, groups)
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]
    np.random.RandomState(SEED).shuffle(train_idx)

    # Safety check: no track shared between train and test
    assert not (set(groups[train_idx]) & set(groups[test_idx])), "Track leakage train/test!"
    assert not (set(groups[train_idx]) & set(groups[val_idx])), "Track leakage train/val!"

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} (split by sign track, no overlap)")

    save_sample_images(y, paths, test_idx)

    print("\nBuilding CNN model...")
    model = build_model(NUM_CLASSES)

    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    print("\nEvaluating on held-out test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print("\nSaving model...")
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump({str(k): v for k, v in CLASS_NAMES.items()}, f, indent=2)

    with open(ACCURACY_PATH, "w") as f:
        f.write(str(round(float(test_accuracy), 4)))

    metrics = {
        "test_accuracy": round(float(test_accuracy), 4),
        "test_loss": round(float(test_loss), 4),
        "test_samples": int(len(X_test)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "num_classes": NUM_CLASSES,
        "epochs_run": len(history.history["loss"]),
        "split_method": "by sign track (no track in both train and test)"
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Test Accuracy: {test_accuracy*100:.2f}% on {len(X_test)} unseen test images")
    print(f"Model: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()