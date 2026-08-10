import numpy as np

from utils.preprocessing import preprocessing_data
from models.neural_network import NeuralNetwork
from utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix

train_file = "../data/train.csv"

X_train, X_test, y_train, y_test = preprocessing_data(train_file)

np.random.seed(42)

network = NeuralNetwork(
  input_size= X_train.shape[1],
  hidden_size= 10,
  learning_rate= 0.01
)

network.train(
  X_train,
  y_train,
  epochs= 1000,
  print_every= 50
)


predictions = network.forward(X_test)

predictions = (predictions >= 0.5).astype(int)


test_accuracy = accuracy(y_test, predictions)
test_precision = precision(y_test, predictions)
test_recall = recall(y_test, predictions)
test_f1 = f1_score(y_test, predictions)

tp, fp, fn, tn = confusion_matrix(y_test, predictions)

print(f"Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {test_precision * 100:.2f}%")
print(f"Recall: {test_recall * 100:.2f}%")
print(f"F1 Score: {test_f1 * 100:.2f}%")

print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("TN:", tn)

