import numpy as np


class NeuralNetwork:
    """From-scratch feedforward neural network with ReLU hidden activations.

    Fixes over the original implementation:
    - Biases are applied in forward() and their gradients computed in backward()
    - Output layer uses linear (regression) or sigmoid (binary classification) — no ReLU leak
    - He initialization for stable training with ReLU
    - initialize_weights_and_biases() no longer mutates the architecture on retrain
    """

    def __init__(self, input_size: int, output_size: int, lr: float, output_activation: str = "linear"):
        assert output_activation in ("linear", "sigmoid"), "output_activation must be 'linear' or 'sigmoid'"
        self.input_neurons = input_size
        self.output_neurons = output_size
        self.lr = lr
        self.output_activation = output_activation
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._hidden_sizes: list[int] = []

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def add_layer(self, neurons: int) -> None:
        self._hidden_sizes.append(neurons)

    def initialize_weights_and_biases(self) -> None:
        sizes = [self.input_neurons] + self._hidden_sizes + [self.output_neurons]
        np.random.seed(42)
        self.biases = [np.zeros((1, l)) for l in sizes[1:]]
        # He initialization: variance = 2/fan_in (suited for ReLU)
        self.weights = [
            np.random.randn(l_prev, l) * np.sqrt(2.0 / l_prev)
            for l_prev, l in zip(sizes[:-1], sizes[1:])
        ]

    # ------------------------------------------------------------------
    # Activations & loss
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _output_fn(self, z: np.ndarray) -> np.ndarray:
        return self._sigmoid(z) if self.output_activation == "sigmoid" else z

    def cost(self, h: np.ndarray, y: np.ndarray) -> float:
        if self.output_activation == "sigmoid":
            h = np.clip(h, 1e-12, 1 - 1e-12)
            return float(-np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))
        return float(np.sum(np.square(h - y)))

    # ------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = X
        n = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(h, w) + b
            h = self._output_fn(z) if i == n - 1 else np.maximum(z, 0)
        return h

    def backward(self, X: np.ndarray, y: np.ndarray) -> tuple[list, list, float]:
        n = len(self.weights)
        d_b = [np.zeros(b.shape) for b in self.biases]
        d_w = [np.zeros(w.shape) for w in self.weights]

        h = X
        hs = [X]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(h, w) + b
            zs.append(z)
            h = self._output_fn(z) if i == n - 1 else np.maximum(z, 0)
            hs.append(h)

        loss = self.cost(hs[-1], y)

        d_z = [np.zeros(z.shape) for z in zs]

        # Output gradient: BCE+sigmoid → ŷ-y (clean combined form); MSE+linear → 2(ŷ-y)
        d_z[-1] = hs[-1] - y if self.output_activation == "sigmoid" else 2 * (hs[-1] - y)
        d_w[-1] = np.dot(hs[-2].T, d_z[-1])
        d_b[-1] = np.sum(d_z[-1], axis=0, keepdims=True)

        for l in range(2, n + 1):
            dz = np.dot(d_z[-l + 1], self.weights[-l + 1].T)
            dz[zs[-l] < 0] = 0  # ReLU derivative
            d_z[-l] = dz
            d_w[-l] = np.dot(hs[-l - 1].T, dz)
            d_b[-l] = np.sum(dz, axis=0, keepdims=True)

        return d_b, d_w, loss

    # ------------------------------------------------------------------
    # Training & inference
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, log_every: int = 50) -> list[float]:
        self.initialize_weights_and_biases()
        history: list[float] = []
        for epoch in range(epochs):
            grad_b, grad_w, loss = self.backward(X, y)
            self.weights = [w - self.lr * gw for w, gw in zip(self.weights, grad_w)]
            self.biases = [b - self.lr * gb for b, gb in zip(self.biases, grad_b)]
            history.append(loss)
            if epoch % log_every == 0:
                print(f"Epoch {epoch:>4} | Cost: {loss:.4f}")
        print(f"Epoch {epochs - 1:>4} | Cost: {history[-1]:.4f}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict(X) >= threshold).astype(int)

    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
        return float(np.mean((y_pred >= threshold).astype(int) == y_true))
