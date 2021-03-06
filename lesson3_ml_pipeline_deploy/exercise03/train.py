import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

with open('params.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

X = np.loadtxt("X.csv")
y = np.loadtxt("y.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=23
)

lr = LogisticRegression(C=params['train']['lr_C'])
lr.fit(X_train.reshape(-1, 1), y_train)

preds = lr.predict(X_test.reshape(-1, 1))
f1 = f1_score(y_test, preds)
print(f"F1 score: {f1:.4f}")
