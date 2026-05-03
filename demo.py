import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_network import NeuralNetwork

# --- Data ----------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target.reshape(-1, 1)   # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Dataset     : Breast Cancer Wisconsin")
print(f"Classes     : {data.target_names[0]} (0) vs {data.target_names[1]} (1)")
print(f"Train/Test  : {len(X_train)} / {len(X_test)} samples | {X_train.shape[1]} features\n")

# --- Model ---------------------------------------------------------------
nn = NeuralNetwork(input_size=X_train.shape[1], output_size=1, lr=0.001, output_activation="sigmoid")
nn.add_layer(64)
nn.add_layer(32)

print(f"Architecture: {X_train.shape[1]} → 64 → 32 → 1  (ReLU hidden, sigmoid output, BCE loss)")
print(f"Learning rate: {nn.lr}\n")

# --- Training loop (manual to track per-epoch accuracy) ------------------
EPOCHS = 300
LOG_EVERY = 50

nn.initialize_weights_and_biases()
history: list[float] = []
train_acc_history: list[float] = []
test_acc_history: list[float] = []

for epoch in range(EPOCHS):
    grad_b, grad_w, loss = nn.backward(X_train, y_train)
    nn.weights = [w - nn.lr * gw for w, gw in zip(nn.weights, grad_w)]
    nn.biases = [b - nn.lr * gb for b, gb in zip(nn.biases, grad_b)]
    history.append(loss)

    train_acc = NeuralNetwork.accuracy(nn.predict(X_train), y_train)
    test_acc = NeuralNetwork.accuracy(nn.predict(X_test), y_test)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    if epoch % LOG_EVERY == 0:
        print(f"Epoch {epoch:>4} | Loss: {loss:.4f} | Train: {train_acc:.3f} | Test: {test_acc:.3f}")

print(f"Epoch {EPOCHS-1:>4} | Loss: {history[-1]:.4f} | Train: {train_acc_history[-1]:.3f} | Test: {test_acc_history[-1]:.3f}")

# --- Metrics -------------------------------------------------------------
y_pred = nn.predict_classes(X_test)
tp = int(np.sum((y_pred == 1) & (y_test == 1)))
tn = int(np.sum((y_pred == 0) & (y_test == 0)))
fp = int(np.sum((y_pred == 1) & (y_test == 0)))
fn = int(np.sum((y_pred == 0) & (y_test == 1)))
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"\n--- Final Results ---")
print(f"Train accuracy : {train_acc_history[-1]*100:.2f}%")
print(f"Test accuracy  : {test_acc_history[-1]*100:.2f}%")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"\nConfusion Matrix (test set):")
print(f"  TP={tp}  FP={fp}")
print(f"  FN={fn}  TN={tn}")

# --- Plot ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("NeuralNetwork — Breast Cancer Classification", fontsize=13, fontweight="bold")

axes[0].plot(history, color="#2563eb", linewidth=1.5)
axes[0].set_title("Training Loss (BCE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary Cross-Entropy")
axes[0].grid(True, linestyle="--", alpha=0.4)

axes[1].plot(train_acc_history, color="#2563eb", linewidth=1.5, label=f"Train ({train_acc_history[-1]:.1%})")
axes[1].plot(test_acc_history, color="#16a34a", linewidth=1.5, label=f"Test ({test_acc_history[-1]:.1%})")
axes[1].set_title("Accuracy over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0.5, 1.05)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to training_results.png")
