import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('D:/1A/ML/titanic.csv')

# Drop unnecessary columns
df = df.drop(['PassengerId', 'Ticket'], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df['Age'] = imputer.fit_transform(df[['Age']]).ravel()
df['Embarked'] = imputer.fit_transform(df[['Embarked']]).ravel()

# Drop Cabin (too many missing values)
df = df.drop('Cabin', axis=1)

# Feature Engineering (Title extraction)
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df = df.drop('Name', axis=1)

# Encoding categorical variables
le = LabelEncoder()
for col in ['Sex', 'Embarked', 'Title']:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Initial train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature importance using Random Forest
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_train, y_train)

importance = pd.DataFrame({
    'Features': X_train.columns,
    'importance': rf_temp.feature_importances_
})

print("\nAll Feature Importance Scores:")
print(importance)

# Drop low importance features
low_features = importance[importance['importance'] < 0.05]['Features']
print("\nDropping Low Importance Features:", list(low_features))

X = X.drop(columns=low_features)

# Re-split after feature selection
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====== MODELS ====== #

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree Result:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Result:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("\nSVM Result:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
print("\nLogistic Regression Result:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

#======FINAL ACCURACY======# 

print("\n===== FINAL ACCURACY =====")
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))
print("SVM:", accuracy_score(y_test, y_pred_svm))
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
