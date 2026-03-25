from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

for k in range(1,6):
    #instantiate learning model(k=1to5)
    classifier = KNeighborsClassifier(n_neighbors=k)

    #fitting the model
    classifier.fit(x_train,y_train)

    #predicting the test set results
    y_pred = classifier.predict(x_test)

    cm = confusion_matrix(y_test,y_pred)
    print(cm)

    accuracy = accuracy_score(y_test,y_pred)*100
    print('Accuracy of our model is equal ' +str(round(accuracy,2))+'%.')
