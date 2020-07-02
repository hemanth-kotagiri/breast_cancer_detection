from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd

# X, y = load_breast_cancer()["data"], load_breast_cancer()["target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

data_df = pd.read_csv("Data/data.csv")

y = data_df["diagnosis"].apply(lambda x: 1 if x == 'M' else 0)
# print(data_df.columns)


features = ['radius_mean', 'texture_mean', 'perimeter_mean',   
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',       
       'symmetry_worst', 'fractal_dimension_worst']

X_train, X_test, y_train, y_test = train_test_split(data_df[features], y, test_size=0.2)


models = [LogisticRegression(random_state=0, max_iter=5000), RandomForestClassifier(), SVC()]

for i,model in enumerate(models, start = 1):
    model.fit(X_train, y_train)
    print(f"Model {i} Accuracy: ", precision_score(y_test, model.predict(X_test)))