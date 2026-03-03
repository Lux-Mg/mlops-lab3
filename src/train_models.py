import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import yaml

params = yaml.safe_load(open("params.yaml"))["train"]

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/base"
os.makedirs(MODEL_DIR, exist_ok=True)

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

X_train = np.load(os.path.join(PROCESSED_DIR, "X_flat.npy"))
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_flat.npy"))
X_train_cnn = np.load(os.path.join(PROCESSED_DIR, "X_cnn.npy"))
X_test_cnn = np.load(os.path.join(PROCESSED_DIR, "X_test_cnn.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

results = {}

# classical ML
rf = RandomForestClassifier(n_estimators=100, random_state=params["random_state"])
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
results["RandomForest"] = accuracy_score(y_test, preds)
joblib.dump(rf, os.path.join(MODEL_DIR, "RandomForest.joblib"))

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
results["LogisticRegression"] = accuracy_score(y_test, preds)
joblib.dump(lr, os.path.join(MODEL_DIR, "LogisticRegression.joblib"))

# CNN
cnn = create_cnn_model((params["img_height"], params["img_width"], 3), params["num_classes"])
cnn.fit(X_train_cnn, tf.keras.utils.to_categorical(y_train),
        validation_data=(X_test_cnn, tf.keras.utils.to_categorical(y_test)),
        epochs=params["epochs"], batch_size=params["batch_size"], verbose=1)
_, acc = cnn.evaluate(X_test_cnn, tf.keras.utils.to_categorical(y_test), verbose=0)
results["CNN"] = acc
cnn.save(os.path.join(MODEL_DIR, "CNN.h5"))

results_clean = {k: float(v) for k, v in results.items()}
with open("metrics.json", "w") as f:
    json.dump(results_clean, f, indent=2)

