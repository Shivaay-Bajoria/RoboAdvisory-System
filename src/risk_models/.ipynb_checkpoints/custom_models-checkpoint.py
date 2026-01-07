import numpy as np

class SoftMaxRegression:
    def __init__(self, learning_rate=0.5, epochs=5000, random_state=42):
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y, num_classes):
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def fit(self, X, y):
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))
        y_encoded = self._one_hot(y, num_classes)

        self.W = np.random.randn(n_features, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))

        for epoch in range(self.epochs):
            logits = np.dot(X, self.W) + self.b
            probs = self._softmax(logits)

            loss = -np.mean(np.sum(y_encoded * np.log(probs + 1e-9), axis=1))

            dW = (1 / n_samples) * np.dot(X.T, (probs - y_encoded))
            db = (1/ n_samples) * np.sum(probs - y_encoded, axis=0, keepdims=True)

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
                logits = np.dot(X, self.W) + self.b
                probs = self._softmax(logits)
                return np.argmax(probs, axis=1)
        
    def predict_proba(self, X):
        logits = np.dot(X, self.W) + self.b
        return self._softmax(logits)

    def accuracy(self, y_true, y_pred):
                return np.mean(y_true == y_pred)