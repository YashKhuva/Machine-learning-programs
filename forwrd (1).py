import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

# 1. Load Data
df = pd.read_csv(r"D:\yash\Auto MPG.csv")

# 2. Data Cleaning
df.replace('?', np.nan, inplace=True)
df['horsepower'] = pd.to_numeric(df['horsepower'])
df = df.dropna()

# 3. Preprocessing
if 'car name' in df.columns:
    df = df.drop(columns=['car name'])

X = df.drop(columns=['mpg'])
y = df['mpg']

# Split into Train and Test (Keep Test strictly for final evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Forward Feature Selection using Cross-Validation
remaining_features = list(X.columns)
selected_features = []
best_score = -np.inf

print("Forward feature selection process (using 5-fold CV on Train set):\n")

while remaining_features:
    scores = [] 

    for feature in remaining_features:
        features_to_test = selected_features + [feature]
        model = LinearRegression()
        
        # Use Cross-Validation on the training set only
        # This ensures our feature selection is robust
        cv_scores = cross_val_score(model, X_train[features_to_test], y_train, cv=5, scoring='r2')
        score = np.mean(cv_scores)
        scores.append((score, feature))

    scores.sort(reverse=True)
    current_best_score, best_feature = scores[0]

    # Termination criteria: only add if it improves the CV score
    if current_best_score > best_score:
        best_score = current_best_score
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f'Added: {best_feature:15} | CV R2 score: {best_score:.4f}')
    else:
        break

# 5. Final Evaluation
final_model = LinearRegression()
final_model.fit(X_train[selected_features], y_train)
test_r2 = r2_score(y_test, final_model.predict(X_test[selected_features]))

print(f'\nFinal selected features: {selected_features}')
print(f'Final R2 score on Test Set: {test_r2:.4f}')