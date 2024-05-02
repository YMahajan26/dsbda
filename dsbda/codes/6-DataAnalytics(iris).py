import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

data = pd.read_csv('6-iris.csv')

print(data.columns)
print("---------------------------------------------------------------")

print(data.shape)
print("---------------------------------------------------------------")

print(data.describe())
print("---------------------------------------------------------------")

print(data.info())
print("---------------------------------------------------------------")

#cleaning Nan Value
#before cleaning conver specoes name to numbers
data['Species'].replace({'Setosa': 0, 'Versicolor': 1, 'Virginica' : 2},inplace=True)
print(data.head(10))
print("-------------------------------------------------------")
data.fillna(data.median(), inplace=True)

#spliting data
x = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = data['Species']

#train and test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 48)

model = GaussianNB()
model.fit(x_train,y_train)

#predection step and will predect Y value
y_pred = model.predict(x_test)
print("\n Predected values in testing data :")
print(y_pred)
print("-------------------------------------------------------")

#Get the accuracy value for how good the training is
print("\nAccuracy value for how good the training is :")
print(model.score(x_train, y_train))
print("Accuracy value for how good the training data is : ", model.score(x_train, y_train)*100,"%") #shows kitna accurate predection hoga
print("-------------------------------------------------------")

#Get the accuracy value for how good the testing is
print("\nAccuracy value for how good the testing is :")
print(model.score(x_test, y_test))
print("Accuracy value for how good the testing data is : ", model.score(x_test, y_test)*100,"%")
print("-------------------------------------------------------")


#########........PART 2 OF QUESTION.........##################

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("--------------------------------------------------------")

# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# print( "TN = ",tn,"\nFP = " ,fp,"\nFN = ",fn,"\nTP = ",tp)

a = accuracy_score(y_test, y_pred)
print("\nAccuracy score is : ",a)

p = precision_score(y_test, y_pred, average = 'weighted')
print("\nPrecision score is : ",p)

r = recall_score(y_test, y_pred, average = 'weighted')
print("\nRecall score is : ",r)

e = 1 - a
print("\nError rate is : ",e)
print("-----------------------------------------------------------------")

cm = classification_report(y_test,y_pred)
print(cm)


#TP - True Positive ( model predicts what he reality is )
#FP - False Positive (model predicts false when the reality is different)


#                              [Predicted]
#                           NO             YES
#  Actual            NO    {TN}            {FP}        FP is also TYPE 1 ERROR
#  values            YES   {FN}            {TP}        FN is also knows as TYPE 2 ERROR


# Recall     = TP / Actual "YES"
# Precision  = TP / Predicted "YES"

# Accuracy   = (TP + TN) / Total of YES + NO
# Error rate = 1 - Accuracy OR (FP + FN) / Total of YES + NO


