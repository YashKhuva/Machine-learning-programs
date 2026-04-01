import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Helper function (avoids warning)
def predict_value(model, value, columns):
    return model.predict(pd.DataFrame([[value]], columns=columns))


# ======== 1. CPU TIME vs DISK I/O =========
data = pd.read_csv("D:/1A/ML/cputime.csv")

x = data.iloc[:, 0:1]
y = data.iloc[:, 1]

print("First 5 rows of x (Disk I/O):")
print(x.head())
print("\nFirst 5 rows of y (CPU time):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_40 = predict_value(model, 40, X_train.columns)

print(f"\nCPU time when Disk I/O = 40: {y_pred_40[0]:.2f}")

plt.scatter(X_train, y_train, color='lightcoral', label='Training data')
plt.plot(X_train, y_pred_train, color='firebrick', label='Regression line')
plt.title('CPU time vs Disk I/O')
plt.xlabel('Disk I/O')
plt.ylabel('CPU time')
plt.legend()
plt.show()


# ======== 2. EXPERIENCE vs SALARY ========
data = pd.read_csv("D:/1A/ML/expvssal.csv")

x = data.iloc[:, 0:1]
y = data.iloc[:, 1]

print("\nFirst 5 rows of x (Experience):")
print(x.head())
print("\nFirst 5 rows of y (Salary):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_11 = predict_value(model, 11, X_train.columns)

print(f"\nPredicted salary for 11 years: ${y_pred_11[0]:,.2f}")

plt.scatter(X_train, y_train, color='lightcoral', label='Training data')
plt.plot(X_train, y_pred_train, color='firebrick', label='Regression line')
plt.title('Experience vs Salary')
plt.xlabel('Experience (years)')
plt.ylabel('Salary ($)')
plt.legend()
plt.show()


# ========== 3. PROFIT vs POPULATION ==========
data = pd.read_csv("D:/1A/ML/linearregressiondataset.csv")

x = data.iloc[:, 0:1]
y = data.iloc[:, 1]

print("\nFirst 5 rows of x (Population):")
print(x.head())
print("\nFirst 5 rows of y (Profit):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nCoefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_pred_20_27 = predict_value(model, 20.27, X_train.columns)

print(f"\nPredicted profit for population 20.27: ${y_pred_20_27[0]:,.2f}")

print("\nModel Evaluation:")
print(f"MAE: ${metrics.mean_absolute_error(y_test, y_pred_test):,.2f}")
print(f"MSE: ${metrics.mean_squared_error(y_test, y_pred_test):,.2f}")
print(f"RMSE: ${np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)):.2f}")

plt.scatter(X_train, y_train, color='lightcoral', label='Training data')
plt.plot(X_train, y_pred_train, color='firebrick', label='Regression line')
plt.title('Profit vs Population')
plt.xlabel('Population (lakhs)')
plt.ylabel('Profit ($)')
plt.legend()
plt.show()
