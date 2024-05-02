'''1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
2. Create a histogram for each feature in the dataset to illustrate the feature distributions. 
3. Create a boxplot for each feature in the dataset. 
4. Compare distributions and identify outliers.'''

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv('10-Iris.csv')
df = df.drop('Id', axis=1)
df.columns = ('SL', 'SW', 'PL', 'PW', 'Species')
#---------------------------------------------------------------------------------------
# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
#---------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display and fill the Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------

# 2. Create a histogram for each feature in the dataset to illustrate the feature distributions. 
fig, axis = plt.subplots(2,2,figsize=(12,8))
fig.suptitle('Histogram of 1 variable')
sns.histplot(ax = axis[0,0], data = df, x='SL', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[0,1], data = df, x='SW', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[1,0], data = df, x='PL', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[1,1], data = df, x='PW', multiple = 'dodge', shrink = 0.8, kde = True)
plt.show()

fig, axis = plt.subplots(2,2,figsize=(12,6))
fig.suptitle('Histogram of 2 variable')
# sns.histplot(ax=axis[0,0], data=df, x='SL', hue='Species',element='poly', shrink=0.8, kde= True)
# sns.histplot(ax=axis[0,1], data = df, x='SW', hue = 'Species', element = 'poly', shrink = 0.8, kde = True)
# sns.histplot(ax=axis[1,0], data = df, x='PL', hue = 'Species', element = 'poly', shrink = 0.8, kde = True)
# sns.histplot(ax=axis[1,1], data = df, x='PW', hue = 'Species', element = 'poly', shrink = 0.8, kde = True)
sns.histplot(ax=axis[0,0], data=df, x='SL', hue='Species', multiple = 'dodge')
sns.histplot(ax=axis[0,1], data = df, x='SW', hue = 'Species', multiple = 'dodge')
sns.histplot(ax=axis[1,0], data = df, x='PL', hue = 'Species', multiple = 'dodge')
sns.histplot(ax=axis[1,1], data = df, x='PW', hue = 'Species', multiple = 'dodge')
plt.show()

# 3. Create a boxplot for each feature in the dataset. 
fig, axis = plt.subplots(2,2 ,figsize=(12,6))
fig.suptitle('Boxplot for each feature in the dataset')
sns.boxplot(ax = axis[0,0], data = df, y='SL')
sns.boxplot(ax = axis[0,1], data = df, y='SW')
sns.boxplot(ax = axis[1,0], data = df, y='PL')
sns.boxplot(ax = axis[1,1], data = df, y='PW')
plt.show()

fig, axis = plt.subplots(2,2,figsize=(12,6))
fig.suptitle('Boxplot for each feature in the dataset')
sns.boxplot(ax = axis[0,0], data = df, y='SL', hue='Species', x='Species')
sns.boxplot(ax = axis[0,1], data = df, y='SW', hue='Species', x='Species')
sns.boxplot(ax = axis[1,0], data = df, y='PL', hue='Species', x='Species')
sns.boxplot(ax = axis[1,1], data = df, y='PW', hue='Species', x='Species')
plt.show()


# 4. Compare distributions and identify outliers.
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.tight_layout()
plt.show()
# sns.pairplot(data=df, hue='Species')
# plt.show()