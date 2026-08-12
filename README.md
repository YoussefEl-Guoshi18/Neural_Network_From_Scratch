# Neural Network From Scratch

A binary classification neural network implemented from scratch using Python and NumPy. The project uses the Titanic dataset to predict whether a passenger survived based on selected passenger information.

The goal of this project was to understand how a neural network works internally rather than relying on a machine learning framework to handle the implementation.

## Overview

The neural network is built from scratch and includes:

- Forward propagation
- Backpropagation
- Gradient descent
- ReLU activation
- Sigmoid activation
- Binary cross-entropy loss
- He weight initialization
- Early stopping
- Classification thresholding
- Confusion matrix evaluation
- Accuracy, precision, recall and F1 score

The project uses NumPy for the neural network calculations and scikit-learn only for splitting the dataset into training and testing sets.

## Dataset

The model uses four features from the Titanic dataset:

- `Pclass`
- `Sex`
- `Age`
- `Fare`

Preprocessing includes:

- Converting `Sex` into a numerical value
- Filling missing `Age` values using the median
- Splitting the data into 80% training and 20% testing data
- Standardising the features using statistics calculated from the training data

## Neural Network Architecture

```text
Input Layer
4 features
    ↓
Hidden Layer
10 neurons
ReLU activation
    ↓
Output Layer
1 neuron
Sigmoid activation
```
## Model Configuration

- Input features: 4
- Hidden neurons: 10
- Hidden activation: ReLU
- Output activation: Sigmoid
- Learning rate: 0.01
- Maximum epochs: 1000
- Weight initialization: He initialization
- Early stopping: Enabled
- Classification threshold: 0.5

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Results

| Metric | Score |
|---|---:|
| Accuracy | 78.21% |
| Precision | 79.66% |
| Recall | 63.51% |
| F1 Score | 70.68% |

### Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN | FP |
| Actual 1 | FN | TP |
