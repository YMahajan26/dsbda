# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    # 1. divide the dataset into training and testing 80%train 20%testing
    # 2. Choose the model (linear regression)
    # 3. Train the model using training data
    # 4. Test the model using testing data
    # 5. Improve the performance of the model
    # Training and testing data
    from sklearn.model_selection import train_test_split
    # Assign test data size 20%
    xtrain, xtest, ytrain, ytest =train_test_split(X,Y,test_size= 0.20, random_state=0)
    # Model selection and training
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model = model.fit(xtrain,ytrain) #Training
    #Testing the model & show its accuracy / Performance
    ypred = model.predict(xtest)

    plt.scatter(ytrain ,model.predict(xtrain),c='blue',marker='o',label='Training data')
    plt.scatter(ytest,ypred ,c='lightgreen',marker='s',label='Test data')
    plt.xlabel('True values')
    plt.ylabel('Predicted')
    plt.title("True value vs Predicted value")
    plt.legend(loc= 'upper left')
    #plt.hlines(y=0,xmin=0,xmax=50)
    plt.plot()
    plt.show()


    from sklearn.metrics import mean_absolute_error
    print('\nMAE:',mean_absolute_error(ytest,ypred))
    print("\nModel Score:",model.score(xtest,ytest))
    print("\n")
#---------------------------------------------------------------------------------------

# Reading dataset
df = pd.read_csv('4b-Boston.csv')
# Display basic information
print('\nInformation of Dataset:\n', df.info)
print('\nShape of Dataset (row x column): ', df.shape)
print('\nColumns Name: ', df.columns)
print('\nTotal elements in dataset:', df.size)
print('\nDatatype of attributes (columns):', df.dtypes)
print('\nFirst 5 rows:\n', df.head())
print('\nLast 5 rows:\n',df.tail())
print('\nAny 5 rows:\n',df.sample(5))
#---------------------------------------------------------------------------------------
# Display Statistical information
print('\nStatistical information of Numerical Columns: \n',df.describe().T)
#---------------------------------------------------------------------------------------
# Display Null values
print('\nTotal Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------

# Feature Engineering - find out most relevant features to predict the output
# output is price of the house in boston housing dataset
# Display correlation matrix
sns.heatmap(df.corr(),annot=True)
plt.show()
# we observed that lstat, ptratio and rm have high correlation with cost of flat (medv)

# before removing outliers
X = df.iloc[:,0:13]
Y = df.iloc[:,-1]
BuildModel(X, Y)

# after removing outliers
x_cols = ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat"]
for _ in x_cols:
    df = RemoveOutlier(df,_)
X = df.iloc[:,0:13]
Y = df.iloc[:,-1]
BuildModel(X, Y)

