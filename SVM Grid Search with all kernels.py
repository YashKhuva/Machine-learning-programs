import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score   
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Target")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling + SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)

# Pipeline (FIXED NAME)
pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

# ---------- CV=5 ----------

# Linear
param_grid_linear = {
    'svm__kernel': ['linear'],
    'svm__C': [0.1, 1, 10]
}

grid_linear = GridSearchCV(pipeline, param_grid_linear, cv=5, scoring='accuracy')
grid_linear.fit(X_train, y_train)

print("LINEAR Kernel Best Params:", grid_linear.best_params_)
print("LINEAR Kernel Accuracy:", grid_linear.best_score_)

# Polynomial
param_grid_poly = {
    'svm__kernel': ['poly'],
    'svm__C': [0.1, 1, 10],
    'svm__degree': [2, 3]
}

grid_poly = GridSearchCV(pipeline, param_grid_poly, cv=5, scoring='accuracy')
grid_poly.fit(X_train, y_train)

print("POLY Kernel Best Params:", grid_poly.best_params_)
print("POLY Kernel Accuracy:", grid_poly.best_score_)

# RBF
param_grid_rbf = {
    'svm__kernel': ['rbf'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': [0.1, 1, 10]
}

grid_rbf = GridSearchCV(pipeline, param_grid_rbf, cv=5, scoring='accuracy')
grid_rbf.fit(X_train, y_train)

print("RBF Kernel Best Params:", grid_rbf.best_params_)
print("RBF Kernel Accuracy:", grid_rbf.best_score_)

# Sigmoid
param_grid_sigmoid = {
    'svm__kernel': ['sigmoid'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': [0.1, 1, 10]
}

grid_sigmoid = GridSearchCV(pipeline, param_grid_sigmoid, cv=5, scoring='accuracy')
grid_sigmoid.fit(X_train, y_train)

print("SIGMOID Kernel Best Params:", grid_sigmoid.best_params_)
print("SIGMOID Kernel Accuracy:", grid_sigmoid.best_score_)

# ---------- CV=10 ----------

# Linear
grid_linear = GridSearchCV(pipeline, param_grid_linear, cv=10, scoring='accuracy')
grid_linear.fit(X_train, y_train)

print("LINEAR (CV=10) Best Params:", grid_linear.best_params_)
print("LINEAR (CV=10) Accuracy:", grid_linear.best_score_)

# Polynomial
grid_poly = GridSearchCV(pipeline, param_grid_poly, cv=10, scoring='accuracy')
grid_poly.fit(X_train, y_train)

print("POLY (CV=10) Best Params:", grid_poly.best_params_)
print("POLY (CV=10) Accuracy:", grid_poly.best_score_)

# RBF
grid_rbf = GridSearchCV(pipeline, param_grid_rbf, cv=10, scoring='accuracy')
grid_rbf.fit(X_train, y_train)

print("RBF (CV=10) Best Params:", grid_rbf.best_params_)
print("RBF (CV=10) Accuracy:", grid_rbf.best_score_)

# Sigmoid (FIXED BLOCK)
param_grid_sigmoid = {
    'svm__kernel': ['sigmoid'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': [0.1, 1, 10]
}

grid_sigmoid = GridSearchCV(pipeline, param_grid_sigmoid, cv=10, scoring='accuracy')
grid_sigmoid.fit(X_train, y_train)

print("SIGMOID (CV=10) Best Params:", grid_sigmoid.best_params_)
print("SIGMOID (CV=10) Accuracy:", grid_sigmoid.best_score_)

# ---------- KNN ----------

knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("KNN Best Parameters:", grid.best_params_)
print("KNN Best Accuracy:", grid.best_score_)
