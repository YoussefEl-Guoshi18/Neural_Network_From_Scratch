import numpy as np

def binary_cross_entropy(y_true, y_pred):
   # avoid log(0) error
  y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) 
  
  return -np.mean(
    y_true * np.log(y_pred) 
    + (1 - y_true) * np.log(1 - y_pred)
    )