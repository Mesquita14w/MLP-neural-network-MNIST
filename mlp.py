import pandas as pd
import numpy as np
import os

DATA_DIR = "./data"

"""
This is the main file of the project, it contains all the MLP class
and some math functions that are used in the MLP class.

I use run code with all the steps in my local IDE (is i mentioned 
earlier, i don't like to use Jupyter Notebooks.), but you can use
whereaver you want.

All the metrics and results of the model are in the repository.

This code just contains the model architecture.
"""

# Load the data:

train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

raw_train = pd.read_csv(train_path).to_numpy()
y_all = raw_train[:, 0].astype(np.int64)
X_all = raw_train[:, 1:].T / 255.0

if os.path.exists(test_path):
    raw_test = pd.read_csv(test_path).to_numpy()
    X_kaggle_test = raw_test.T / 255.0  # Shape (784, N_test)

# Math functions and initialization:

def relu(Z):
    return np.maximum(0.0, Z)

def relu_deriv(Z):
    return (Z > 0.0).astype(np.float64)

def softmax(Z):
    # Numerical stability by subtracting the maximum per column:
    shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

def cross_entropy_loss(Y_hat, Y, eps=1e-12):
    m = Y.shape[1]
    Y_hat_clipped = np.clip(Y_hat, eps, 1.0 - eps)
    return -np.sum(Y * np.log(Y_hat_clipped)) / m

def one_hot_encode(y, num_classes=10):
    # y has shape (N,) with integers from 0 to 9 -> returns (10, N)
    one_hot = np.zeros((num_classes, y.shape[0]), dtype=np.float64)
    one_hot[y, np.arange(y.shape[0])] = 1.0
    return one_hot

# Model architecture and training:

class MLP:
    def __init__(self, layer_dims, learning_rate=0.05, seed=42):
        """
        layer_dims: ex. [784, 128, 64, 10]
        """
        rng = np.random.default_rng(seed)
        self.layer_dims = layer_dims
        self.lr = learning_rate
        self.num_layers = len(layer_dims) - 1
        self.params = {}

        for l in range(1, self.num_layers + 1):
            n_in = layer_dims[l - 1]
            n_out = layer_dims[l]
            # He initialization for hidden layers; moderate scale for output
            scale = np.sqrt(2.0 / n_in) if l < self.num_layers else np.sqrt(1.0 / n_in)
            self.params[f"W{l}"] = rng.normal(0.0, scale, size=(n_out, n_in))
            self.params[f"b{l}"] = np.zeros((n_out, 1), dtype=np.float64)

    def forward(self, X):
        cache = {"A0": X}
        A = X

        for l in range(1, self.num_layers):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = W @ A + b
            A = relu(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        # Last layer: linear + softmax
        L = self.num_layers
        ZL = self.params[f"W{L}"] @ A + self.params[f"b{L}"]
        AL = softmax(ZL)
        cache[f"Z{L}"] = ZL
        cache[f"A{L}"] = AL
        return AL, cache

    def backward(self, Y, cache):
        grads = {}
        m = Y.shape[1]
        L = self.num_layers

        AL = cache[f"A{L}"]
        dZ = (AL - Y) / m

        for l in range(L, 0, -1):
            A_prev = cache[f"A{l - 1}"]
            W = self.params[f"W{l}"]

            grads[f"dW{l}"] = dZ @ A_prev.T
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                dA_prev = W.T @ dZ
                Z_prev = cache[f"Z{l - 1}"]
                dZ = dA_prev * relu_deriv(Z_prev)

        return grads

    def step(self, grads):
        for l in range(1, self.num_layers + 1):
            self.params[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * grads[f"db{l}"]

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, epochs=15, batch_size=128, seed=42):
        rng = np.random.default_rng(seed)
        m = X_train.shape[1]
        history = {"train_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(m)
            X_shuf = X_train[:, perm]
            Y_shuf = Y_train[:, perm]

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                Xb = X_shuf[:, start:end]
                Yb = Y_shuf[:, start:end]

                Y_hat, cache = self.forward(Xb)
                grads = self.backward(Yb, cache)
                self.step(grads)

            train_hat, _ = self.forward(X_train)
            t_loss = cross_entropy_loss(train_hat, Y_train)
            t_acc = self.accuracy(X_train, Y_train)
            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)

            msg = f"Epoch {epoch:02d}/{epochs:02d} - loss: {t_loss:.4f} - acc: {t_acc:.4f}"
            if X_val is not None and Y_val is not None:
                v_acc = self.accuracy(X_val, Y_val)
                history["val_acc"].append(v_acc)
                msg += f" - val_acc: {v_acc:.4f}"
            print(msg)

        return history

    def predict_proba(self, X):
        AL, _ = self.forward(X)
        return AL

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=0)

    # Metrics and evaluation:

    def accuracy(self, X, Y):
        preds = self.predict(X)
        labels = np.argmax(Y, axis=0)
        return np.mean(preds == labels)

    def confusion_matrix_np(y_true, y_pred, num_classes=10):
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

# Train and test split:

def train_val_split(X, y, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    m = X.shape[1]
    perm = rng.permutation(m)

    val_size = int(m * val_ratio)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    X_train, y_train = X[:, train_idx], y[train_idx]
    X_val, y_val = X[:, val_idx], y[val_idx]
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = train_val_split(X_all, y_all, val_ratio=0.1, seed=42)

Y_train = one_hot_encode(y_train, num_classes=10)
Y_val = one_hot_encode(y_val, num_classes=10)

print(f"Training set: {X_train.shape[1]} examples")
print(f"Validation set: {X_val.shape[1]} examples")

# Fit the model:

# Instantiate the MLP (784 -> 128 -> 64 -> 10)
mlp = MLP(layer_dims=[784, 128, 64, 10], learning_rate=0.05, seed=42)

# Execute the fit
print("Starting training...\n")
history = mlp.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val, epochs=20, batch_size=128, seed=42)

# Evaluate the model on the validation set:

def classification_metrics_np(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    tp = np.diag(cm).astype(np.float64)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": np.mean(f1)
    }

y_val_pred = mlp.predict(X_val)
metrics = classification_metrics_np(y_val, y_val_pred)

print("--- CONFUSION MATRIX (Lines: True | Columns: Predicted) ---")
print(metrics["confusion_matrix"])

print("\n--- METRICS BY DIGIT ---")
print("Digit | Precision | Recall   | F1-Score")
print("-" * 42)
for d in range(10):
    print(f"   {d}   |   {metrics['precision'][d]:.4f}  |  {metrics['recall'][d]:.4f}  |  {metrics['f1'][d]:.4f}")

print(f"\nMacro F1-Score on the validation set: {metrics['macro_f1']:.4f}")