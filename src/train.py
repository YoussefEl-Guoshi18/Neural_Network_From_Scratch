import numpy as np
import matplotlib.pyplot as plt

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

losses = network.train(
  X_train,
  y_train,
  epochs= 1000,
  print_every= 50
)

plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("../results/training_loss.png")
plt.close()


predictions = network.forward(X_test)

predictions = (predictions >= 0.5).astype(int)


test_accuracy = accuracy(y_test, predictions)
test_precision = precision(y_test, predictions)
test_recall = recall(y_test, predictions)
test_f1 = f1_score(y_test, predictions)

tp, fp, fn, tn = confusion_matrix(y_test, predictions)

cm = np.array([
  [tn, fp],
  [fn, tp]
])

plt.figure()
plt.imshow(cm)
plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
plt.yticks([0, 1], ["Actual 0", "Actual 1"])
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

for i in range(2):
  for j in range(2):
    value = cm[i, j]

    if value < cm.max() / 2:
      text_colour = "white"
    else:
      text_colour = "black"

    plt.text(j, i, value, ha="center",va="center", color=text_colour)

plt.savefig("../results/confusion_matrix.png")
plt.close()

print(f"Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {test_precision * 100:.2f}%")
print(f"Recall: {test_recall * 100:.2f}%")
print(f"F1 Score: {test_f1 * 100:.2f}%")

print("TP:", tp)
print("FP:", fp)
print("FN:", fn)
print("TN:", tn)

