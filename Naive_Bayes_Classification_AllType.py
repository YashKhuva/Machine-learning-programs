import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes  import GaussianNB,MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,Binarizer

data=load_iris()
X=data.data
y=data.target

df = pd.DataFrame(data.data, columns=data.feature_names)

df.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

print("dataset split complete")
print("Traing samples:",X_train.shape[0])
print("testing samples:,",X_test.shape[0])

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print("GaussianNb Accuracy:", accuracy_score(y_test, y_pred_gnb))

# Multinomial Naive Bayes
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mnb = MultinomialNB()
mnb.fit(X_train_scaled, y_train)
y_pred_mnb = mnb.predict(X_test_scaled)
print("MultinomialNB Accuraccy:", accuracy_score(y_test, y_pred_mnb))

# Bernoulli Naive Bayes
binarizer = Binarizer(threshold=0.0)
X_train_binary = binarizer.fit_transform(X_train)
X_test_binary = binarizer.transform(X_test)

bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train)
y_pred_bnb = bnb.predict(X_test_binary)
print("BernoulliNB Acuracy:", accuracy_score(y_test, y_pred_bnb))     

# Complement Naive Bayes
cnb = ComplementNB()
cnb.fit(X_train_scaled, y_train)
y_pred_cnb = cnb.predict(X_test_scaled)
print("ComplementNB_accuracy:", accuracy_score(y_test, y_pred_cnb))


