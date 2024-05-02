 # Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df  = pd.read_csv('4-HousingData.csv')

print(df.head(10))
print(df.columns)

#seperate independent (x) and dependent values (y)
print("------------------------------------------------------------------------------")
x = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']]
y = df[['MEDV']]

# print(x.head(10))
# print(y.head(10))
#reomove all the NaN values from the datagram as it wont execute regression
#THIS IS GIVING ERROR (BECAUSE REMOVING THE COLUMN GIVES IMPROPER VALUE TO DIVIDE TO 25%)
# print("---------------------------------------------------------")
# print(x.isnull().sum())
# x.dropna(inplace=True)
# y.dropna(inplace=True)
# print("---------------------------------------------------------")
# print(x.isnull().sum())
# print("---------------------------------------------------------")
# print(x.isnull().sum())
##################OR##############################

x = x.fillna(x.mean())
y = y.fillna(y.mean())

print(x.isnull().sum())
print("----------------------------")
print(y.isnull().sum())

#While traing model we split dataset 75% training and 25% testing
#testsize matlab its will divide according to the no. (0.25 = 25% testing) and randomState matlab it will split or take random data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#creating a model
model = LinearRegression()
model.fit(x_train, y_train)

#y predect beacuse y = MDEV (Jitna testing data that uske liye "y" ki values (PREDECTION VALUES)
y_pred = model.predict(x_test)
print("\n Predected values in testing data :")
print(y_pred)

#Get the accuracy value for how good the training is
print("\nAccuracy value for how good the training is :")
print(model.score(x_train, y_train))
print("Accuracy value for how good the training data is : ", model.score(x_train, y_train)*100,"%") #shows kitna accurate predection hoga

#Get the accuracy value for how good the testing is
print("\nAccuracy value for how good the testing is :")
print(model.score(x_test, y_test))
print("Accuracy value for how good the testing data is : ", model.score(x_test, y_test)*100,"%")

#get the error (Between actual value(y_test) and predected value(y_pred)
# A lower MSE indicates better model performance.
print('\nMean squared error-')
print(np.sqrt(mean_squared_error(y_test, y_pred)))

#This will create a scatter plot where the x-axis represents the actual prices (y_test) and the y-axis represents the predicted prices (y_pred).
#The black dashed line represents perfect predictions.
#Points closer to this line indicate better predictions by the model.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)#This line plots a diagonal line from the minimum to the maximum values of y_test.
#  The color is set to black ('k') with a dashed line style ('--'), and the line width is set to 4.
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()