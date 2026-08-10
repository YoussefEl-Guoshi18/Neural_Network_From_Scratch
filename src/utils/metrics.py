import numpy as np

def confusion_matrix(y_true, y_pred):
  tp = np.sum((y_true == 1) & (y_pred == 1))
  fp = np.sum((y_true == 0) & (y_pred == 1))
  tn = np.sum((y_true == 0) & (y_pred == 0))
  fn = np.sum((y_true == 1) & (y_pred == 0))

  return tp, fp, fn, tn


def accuracy(y_true, y_pred):
  return (y_pred == y_true).mean()

def precision(y_true, y_pred):
  tp, fp, fn, tn = confusion_matrix(y_true, y_pred)

  if tp + fp == 0:
    return 1

  return tp / (fp + tp)

def recall(y_true, y_pred):
  tp, fp, fn, tn = confusion_matrix(y_true, y_pred)

  if tp + fn == 0:
    return 1

  return tp/ (tp + fn)

def f1_score(y_true, y_pred):
  p = precision(y_true, y_pred)
  r = recall(y_true, y_pred)

  if p + r == 0:
    return 1

  return 2 * (p * r) / (p + r)