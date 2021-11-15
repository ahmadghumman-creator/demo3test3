import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score
import json
import matplotlib.pyplot as plt

titanic_df = pd.read_csv('data_features.csv')

X = titanic_df.drop(["Survived"], 1)
Y = titanic_df["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.shape)

print(X_test.shape)

DT = DecisionTreeClassifier(criterion='gini')

print(type(DT))

DT.fit(X_train,Y_train)

Y_pred = DT.predict(X_test)

#Predictions with critirion "Entropy"
print("Accuracy: ", accuracy_score(Y_test,Y_pred))
print("Precision: ", precision_score(Y_test,Y_pred))
print("Recall: ", f1_score(Y_test,Y_pred))

acc = str(accuracy_score(Y_test,Y_pred))
pre = str(precision_score(Y_test,Y_pred))
f1s = str(f1_score(Y_test,Y_pred))

X = ['Precision','Recall','f1-score']
Res = [precision_score(Y_test,Y_pred), accuracy_score(Y_test,Y_pred), f1_score(Y_test,Y_pred)]

plt.figure()
# plot the linear data and the exponential data
plt.plot(X, Res, '--o')
plt.xlabel("Measure")
plt.ylabel("Score")
plt.title('Measures')
plt.savefig('results.png')

# Now print to file
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "precision": pre, "f1-score":f1s}, outfile)
outfile.close()
