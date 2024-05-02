# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv('9-titanic.csv')
#---------------------------------------------------------------------------------------
# # Display basic information
# print('Information of Dataset:\n', df.info)
# print('Shape of Dataset (row x column): ', df.shape)
# print('Columns Name: ', df.columns)
# print('Total elements in dataset:', df.size)
# print('Datatype of attributes (columns):', df.dtypes)
# print('First 5 rows:\n', df.head().T)
# print('Last 5 rows:\n',df.tail().T)
# print('Any 5 rows:\n',df.sample(5).T)
# #---------------------------------------------------------------------------------------
# # Display Statistical information
# print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display and fill the Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
# df['Age'].fillna(df['Age'].median(), inplace=True)
df.fillna({'Age': df['Age'].median()}, inplace=True)
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
#One variable
fig, axes = plt.subplots(1,2 ,figsize=(12,6))
sns.boxplot(data = df, y ='Age', ax=axes[0])
sns.boxplot(data = df, y ='Fare', ax=axes[1])
plt.show()

# # Two variables
fig, axes = plt.subplots(1,3, sharey=True,figsize=(12,6))
sns.boxplot(data = df, x='Sex', y ='Age', hue = 'Sex', ax=axes[0])
sns.boxplot(data = df, x='Pclass', y ='Age', hue = 'Pclass', ax=axes[1])
sns.boxplot(data = df, x='Survived', y ='Age', hue = 'Survived', ax=axes[2])
plt.show()

# Two variables
fig, axes = plt.subplots(1,3, sharey=True,figsize=(12,6))

sns.boxplot(data = df, x='Sex', y ='Fare', hue = 'Sex', ax=axes[0], log_scale = True)
sns.boxplot(data = df, x='Pclass', y ='Fare', hue = 'Pclass', ax=axes[1], log_scale = True)
sns.boxplot(data = df, x='Survived', y ='Fare', hue = 'Survived', ax=axes[2], log_scale = True)
plt.show()

#three variables
fig, axes = plt.subplots(1,2, sharey=True,figsize=(12,6))
sns.boxplot(data = df, x='Sex', y ='Age', hue = 'Survived', ax=axes[0])
sns.boxplot(data = df, x='Pclass', y ='Age', hue = 'Survived', ax=axes[1])
plt.show()

fig, axes = plt.subplots(1,2, sharey=True,figsize=(12,6))
sns.boxplot(data = df, x='Sex', y ='Fare', hue = 'Survived', ax=axes[0], log_scale = True)
sns.boxplot(data = df, x='Pclass', y ='Fare', hue = 'Survived', ax=axes[1], log_scale = True)
plt.show()



'''f we want to see the box plots of for age of passengers of both genders, along with the information about whether or not they survived, 
we can pass the survived as value to the hue parameter.
We can also see the distribution of the passengers who survived. For instance, we can see that among the male passengers, on average more
younger people survived as compared to the older ones. Similarly, we can see that the variation among the age of female passengers who 
did not survive is much greater than the age of the surviving female passengers.'''