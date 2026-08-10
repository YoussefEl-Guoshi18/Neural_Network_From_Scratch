import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def select_features(data):

    data = data.copy()

    data["Sex"] = data["Sex"].map({
        "male": 1,
        "female": 0
    })

    data["Age"] = data["Age"].fillna(data["Age"].median())
    
    features = ["Pclass", "Sex", "Age", "Fare"]

    X = data[features].to_numpy()
    y = data["Survived"].to_numpy().reshape(-1, 1)

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state= random_state
    ) 

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    return X_train, X_test

def preprocessing_data(filepath):
    
    data = load_data(filepath)

    X, y = select_features(data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test
