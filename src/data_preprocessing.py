import os
import numpy as np
import yaml
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["train"]

PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

print("Loading Fashion MNIST (Zalando) dataset...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = fashion_mnist.load_data()

# Combine all data for custom train/test split
X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)  # (70000, 28, 28)
y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)   # (70000,)

# Subsample if max_samples is set
max_samples = params.get("max_samples", len(X_all))
if max_samples < len(X_all):
    rng = np.random.RandomState(params["random_state"])
    idx = rng.choice(len(X_all), max_samples, replace=False)
    X_all = X_all[idx]
    y_all = y_all[idx]

print(f"Using {len(X_all)} images, {len(np.unique(y_all))} classes")

# Convert grayscale (28,28) -> RGB (28,28,3) for CNN compatibility
X_rgb = np.stack([X_all, X_all, X_all], axis=-1).astype(np.float32) / 255.0

# Save raw normalized RGB arrays for augmentation step
np.save(os.path.join(RAW_DIR, "X_raw.npy"), X_rgb)
np.save(os.path.join(RAW_DIR, "y_raw.npy"), y_all)

# Flatten for classical ML (RF, LR)
X_flat = X_all.reshape(X_all.shape[0], -1).astype(np.float32) / 255.0

# Train/test split
X_train_flat, X_test_flat, y_train, y_test = train_test_split(
    X_flat, y_all,
    test_size=params["test_size"],
    random_state=params["random_state"]
)
X_train_cnn, X_test_cnn, _, _ = train_test_split(
    X_rgb, y_all,
    test_size=params["test_size"],
    random_state=params["random_state"]
)

np.save(os.path.join(PROCESSED_DIR, "X_flat.npy"), X_train_flat)
np.save(os.path.join(PROCESSED_DIR, "X_test_flat.npy"), X_test_flat)
np.save(os.path.join(PROCESSED_DIR, "X_cnn.npy"), X_train_cnn)
np.save(os.path.join(PROCESSED_DIR, "X_test_cnn.npy"), X_test_cnn)
np.save(os.path.join(PROCESSED_DIR, "y.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

print("Preprocessing complete. Saved to data/processed/")
