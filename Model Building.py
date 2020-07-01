from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import precision_score
import numpy as np
import pandas as pd


X, y = load_breast_cancer()["data"], load_breast_cancer()["target"]

model = LogisticRegression(random_state=0, max_iter = 5000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

print(precision_score(y_test, model.predict(X_test)))
