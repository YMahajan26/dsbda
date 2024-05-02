# logistic regression

# The logistic regression model you've trained is predicting whether a user will make a purchase based on their characteristics.
# In your dataset, the dependent variable Purchased indicates whether a user has purchased the product (1) or not (0).
# So, the model is predicting whether a user will make a purchase
# (Purchased = 1) or not (Purchased = 0) based on the independent variables such as User ID, Gender, Age, and EstimatedSalary.

#in this question Example. you are give some names and you need to "classify" is that a male or a female
#logistic regression is a classification algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df = pd.read_csv('5-Social_Network_Ads.csv')
print(df.head(10))

print("-------------------------------------------------------")
print(df.columns)
print("-------------------------------------------------------")

# category to quantitative (REASON BELOW)
df['Gender'].replace({'Male': 0, 'Female': 1},inplace=True)
print(df.head(10))
print("-------------------------------------------------------")

#spliting data
x = df[['User ID', 'Gender', 'Age', 'EstimatedSalary']]
y = df[['Purchased']]

#remove Nan values (this data set has no nan values so leave it as it is)
print(x.isnull().sum())
print("-------------------------------------------------------")
# x = x.fillna(x.mean())
# y = y.fillna(y.mean())

#split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=78)

#train the model
# model = LogisticRegression()
# model.fit(x_train, y_train)

#ValueError: could not convert string to float: 'Female'.......U will get error like this
#because all the calculations are based on numbers and Male and female are strings
# need to convert catergorical to numerical(quantitative) Male = 0 and female = 1

# df['Gender'].replace({'Male': 0, 'Female': 1} ,inplace=True)
# print(df.head())

#use this code above the spliting data part

#train the model(NEW)
model = LogisticRegression()
model.fit(x_train, y_train)


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



#########.......PART 2 OF QUESTION....################


cm = confusion_matrix(y_test, y_pred)
print(cm)
print("--------------------------------------------------------")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print( "TN = ",tn,"\nFP = " ,fp,"\nFN = ",fn,"\nTP = ",tp)

a = accuracy_score(y_test, y_pred)
print("\nAccuracy score is : ",a)

p = precision_score(y_test, y_pred)
print("\nPrecision score is : ",p)

r = recall_score(y_test, y_pred)
print("\nRecall score is : ",r)

e = 1 - a
print("\nError rate is : ",e)


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


