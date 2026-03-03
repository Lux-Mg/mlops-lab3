import os
import numpy as np
import yaml
import tensorflow as tf
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["train"]

RAW_DIR = "data/raw"
AUGMENTED_DIR = "data/augmented"

os.makedirs(AUGMENTED_DIR, exist_ok=True)

print("Loading raw preprocessed data for augmentation...")
X_raw = np.load(os.path.join(RAW_DIR, "X_raw.npy"))  # (N, 28, 28, 3) float32
y_raw = np.load(os.path.join(RAW_DIR, "y_raw.npy"))  # (N,)

print(f"Applying augmentation to {len(X_raw)} images...")

# Augmentation layers
augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Apply augmentation in batches
BATCH_SIZE = 1000
X_aug_list = []

for i in range(0, len(X_raw), BATCH_SIZE):
    batch = tf.convert_to_tensor(X_raw[i:i + BATCH_SIZE])
    aug_batch = augmenter(batch, training=True).numpy()
    aug_batch = np.clip(aug_batch, 0.0, 1.0).astype(np.float32)
    X_aug_list.append(aug_batch)
    print(f"  Augmented {min(i + BATCH_SIZE, len(X_raw))}/{len(X_raw)} images")

X_aug = np.concatenate(X_aug_list, axis=0)
y_aug = y_raw.copy()

# Combine original + augmented
X_combined = np.concatenate([X_raw, X_aug], axis=0)
y_combined = np.concatenate([y_raw, y_aug], axis=0)

print(f"Total images after augmentation: {len(X_combined)}")

# Flatten for classical ML
X_flat_combined = X_combined.reshape(X_combined.shape[0], -1)

# Train/test split
X_train_flat, X_test_flat, y_train, y_test = train_test_split(
    X_flat_combined, y_combined,
    test_size=params["test_size"],
    random_state=params["random_state"]
)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
    X_combined, y_combined,
    test_size=params["test_size"],
    random_state=params["random_state"]
)

np.save(os.path.join(AUGMENTED_DIR, "X_flat.npy"), X_train_flat)
np.save(os.path.join(AUGMENTED_DIR, "y.npy"), y_train)
np.save(os.path.join(AUGMENTED_DIR, "X_test_flat.npy"), X_test_flat)
np.save(os.path.join(AUGMENTED_DIR, "y_test.npy"), y_test)
np.save(os.path.join(AUGMENTED_DIR, "X_cnn.npy"), X_train_cnn)
np.save(os.path.join(AUGMENTED_DIR, "y_cnn.npy"), y_train_cnn)
np.save(os.path.join(AUGMENTED_DIR, "X_test_cnn.npy"), X_test_cnn)
np.save(os.path.join(AUGMENTED_DIR, "y_test_cnn.npy"), y_test_cnn)

print("Augmentation complete. Saved to data/augmented/")
