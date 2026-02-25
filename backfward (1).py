import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df = pd.read_csv(r"D:\yash\Auto MPG.csv")

df.replace('?', np.nan, inplace=True)
df['horsepower'] = pd.to_numeric(df['horsepower'])
df = df.dropna()


if 'car name' in df.columns:
    df = df.drop(columns=['car name'])

X = df.drop(columns=['mpg'])
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

selected_features = list(X.colums)

model = LinearRegression()
model.fit(X_train[features_to_test], y_train)

y_pred = model.predict(X_test[selected_features])
best_score = r2_score(Y_test, y_pred)

print("Backward feture elimianation process:\n")
print("Inintial R2 score (All feature): {best_score:4f}\n")

while len(selected_feature) > 1:
    score = []

    for feture in selected_features:
        features_to_test = selected_features.copy()
        features_to_test.remove(feature)

        model  = LinearRegression()
        model.fit(X_train[features_to_test], y_train)
        y_pred = model.predict(X_test[selected_features])
        best_score = r2_score(Y_test, y_pred)
        

        
