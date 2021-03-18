from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# splitting data into features and targets
X = iris.data
y = iris.target

# splitting data into training and testing, 20% data for training test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# building the model
model = svm.SVC()

# training the model
model.fit(X_train, y_train)

# getting the model predictions and getting accuracy result
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

# printing the predictions and alll
print(f'Prediction: {predictions}')
print(f'Actual: {y_test}')
print(acc)