# logistic regression .
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------
def RemoveOutlier(df,var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3+1.5*IQR, Q1-1.5*IQR
    print("Highest allowed in variable:", var, high)
    print("lowest allowed in variable:", var, low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print('Total outliers in:',var,':',count)
    df = df[((df[var] >= low) & (df[var] <= high))]
    return df
#---------------------------------------------------------------------------------------
def BuildModel(X, Y):
    # Training and testing data
    from sklearn.model_selection import train_test_split
    # Assign test data size 20%
    xtrain, xtest, ytrain, ytest =train_test_split(X,Y,test_size= 0.25, random_state=13)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver = 'lbfgs')
    model = model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    # print("\nScore: ", model.score(xtest,ytest))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ytest, ypred)
    sns.heatmap(cm, annot=True)
    plt.show()
    from sklearn.metrics import classification_report
    print(classification_report(ytest, ypred))

# Reading dataset
df = pd.read_csv('5-Social_Network_Ads.csv')

# Display basic information
print('\nInformation of Dataset:\n', df.info)
print('\nShape of Dataset (row x column): ', df.shape)
print('\nColumns Name: ', df.columns)
print('\nTotal elements in dataset:', df.size)
print('\nDatatype of attributes (columns):', df.dtypes)
print('\nFirst 5 rows:\n', df.head())
print('\nLast 5 rows:\n',df.tail())
print('\nAny 5 rows:\n',df.sample(5))
df = df.drop('User ID', axis=1)
df.columns = ['Gender', 'Age', 'Salary', 'Purchased']
#---------------------------------------------------------------------------------------
# Display Statistical information
print('\nStatistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display Null values
print('\nTotal Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Label encoding method
df['Gender']=df['Gender'].astype('category')
df['Gender']=df['Gender'].cat.codes
# Display correlation matrix
sns.heatmap(df.corr(),annot=True)
plt.show()
#---------------------------------------------------------------------------------------
# Choosing input and output variables from correlation matrix
X = df[['Age','Salary']]
Y = df['Purchased']
BuildModel(X, Y)
# #---------------------------------------------------------------------------------------
# Checking and removing outliers
fig, axes = plt.subplots(1,2)
sns.boxplot(data = df, x ='Age', ax=axes[0])
sns.boxplot(data = df, x ='Salary', ax=axes[1])
fig.tight_layout()
plt.show()
df = RemoveOutlier(df, 'Age')
df = RemoveOutlier(df, 'Salary')
# You can use normalization method to improve the score
# salary -> high range
# age -> low range
# Normalization will smoothe both range salary and age

# Choosing input and output variables from correlation matrix
X = df[['Age','Salary']]
Y = df['Purchased']
# Checking model score after removing outliers
BuildModel(X, Y)

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
