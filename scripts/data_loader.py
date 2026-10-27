
import os
from pathlib import Path

# Đặt cache dataset trong thư mục dự án trên ổ E
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ["KERAS_HOME"] = str(PROJECT_DIR / ".keras")
os.environ["KERAS_HOME"] = str(PROJECT_DIR / ".keras")

import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    X = X.reshape(-1, 784).astype("float32") / 255.0

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Classes:", np.unique(y))
    print("Project directory:", PROJECT_DIR)

    return X, y


if __name__ == "__main__":
    X, y = load_mnist()
