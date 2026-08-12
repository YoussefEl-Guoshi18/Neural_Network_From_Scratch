import numpy as np
from .activations import relu, relu_derivative, sigmoid
from .loss import binary_cross_entropy


class NeuralNetwork:

  def __init__(self, input_size, hidden_size, learning_rate=0.01):

    self.learning_rate = learning_rate

    self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
    self.b1 = np.zeros((1, hidden_size))

    self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2 / hidden_size)
    self.b2 = np.zeros((1, 1))


  def forward(self, X):

    self.X = X

    self.Z1 = X @ self.W1 + self.b1  
    self.A1 = relu(self.Z1)

    self.Z2 = self.A1 @ self.W2 + self.b2
    self.A2 = sigmoid(self.Z2)

    return self.A2

  def backward(self, Y):

    m = self.X.shape[0]

    dZ2 = self.A2 - Y

    dW2 = (self.A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ self.W2.T
    dZ1 = dA1 * relu_derivative(self.Z1)

    dW1 = (self.X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    self.W1 -= self.learning_rate * dW1
    self.b1 -= self.learning_rate * db1

    self.W2 -= self.learning_rate * dW2
    self.b2 -= self.learning_rate * db2

  def train(self, X, Y, epochs, print_every=50, tol=0.001, patience=5):
    losses = []

    best_loss = float("inf")
    patience_count = 0

    for epoch in range(epochs):

      self.forward(X)

      loss = binary_cross_entropy(Y, self.A2)
      losses.append(loss)

      self.backward(Y)

      if epoch % print_every == 0:
        print(f"Epoch {epoch}, loss {loss:.4f}")

      if best_loss - loss < tol:
        patience_count += 1 
      else:
        patience_count = 0
        best_loss = loss

      if patience_count >= patience:
        print(f"Converged at epoch {epoch}")
        break

    return losses