# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv('8-titanic.csv')
#---------------------------------------------------------------------------------------
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
print('\nStatistical information of Numerical Columns: \n',df.describe())
# ---------------------------------------------------------------------------------------
# Display and fill the Null values
print('\nTotal Number of Null Values in Dataset:', df.isna().sum())
# df['Age'].fillna(df['Age'].median(), inplace=True)
df.fillna({'Age':df['Age'].median()}, inplace=True)

print('\nTotal Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Single variable histogram

fig, axis = plt.subplots(1,3, figsize=(12,6))
sns.histplot(ax = axis[0], data = df, x='Sex', hue = 'Sex', multiple = 'dodge')
sns.histplot(ax = axis[1], data = df, x='Pclass', hue = 'Pclass',multiple = 'dodge')
sns.histplot(ax = axis[2], data = df, x='Survived', hue = 'Survived', multiple = 'dodge')
plt.show()
# Single variable histogram
fig, axis = plt.subplots(1,2, figsize=(12,6))
sns.histplot(ax = axis[0], data = df, x='Age', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[1], data = df, x='Fare', multiple = 'dodge', shrink = 0.8, kde = True)
plt.show()
# Two variable histogram

fig, axis = plt.subplots(2,2, figsize=(12,6))
sns.histplot(ax = axis[0,0], data = df, x='Age', hue = 'Sex', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[0,1], data = df, x='Fare', hue = 'Sex', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,0], data=df, x='Age', hue = 'Survived', multiple = 'dodge',shrink=0.8, kde= True)
sns.histplot(ax = axis[1,1], data=df, x='Fare', hue='Survived', multiple='dodge', shrink=0.8, kde = True)
plt.show()
# Two variable histogram
fig, axis = plt.subplots(2,2, figsize=(12,6))
sns.histplot(ax=axis[0,0], data=df, x='Sex', hue='Survived', multiple= 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax=axis[0,1], data=df, x='Pclass', hue='Survived', multiple='dodge', shrink=0.8, kde= True)
sns.histplot(ax=axis[1,0], data=df, x='Age', hue='Survived', multiple='dodge', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,1], data=df, x='Fare', hue='Survived', multiple='dodge', shrink = 0.8, kde = True)
plt.show()


# # optional plots.......
# # Joint Plot Type-1
# sns.jointplot(x='Age', y='Fare', data=df)
# plt.show()

# # Box Plot
# sns.boxplot(x='Sex', y='Age', data=df, hue="Survived")
# plt.show()

# # Violin Plot
# sns.violinplot(x='Sex', y='Age', data=df, hue='Survived')
# plt.show()

# # Strip Plot
# sns.stripplot(x='Sex', y='Age', data=df, jitter=True, hue='Survived')

# # Matrix Plots
# # Heat Map
# dataset = sns.load_dataset('titanic')
# dataset.head()
# dataset.dropna(inplace=True)
# dataset = pd.get_dummies(dataset)
# corr = dataset.corr()
# sns.heatmap(corr)
# plt.show()

# ---------------------------------------
# Drop non-numeric columns before generating the correlation heatmap
# titanic = sns.load_dataset('titanic')
# print(titanic)
numeric_titanic = df.select_dtypes(include=['float64', 'int64'])

# Heatmap to visualize correlations between numerical variables
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_titanic.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Titanic Dataset')
plt.show()


# Histogram to visualize the distribution of ticket prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'], bins=20, kde=True)
plt.title('Distribution of Ticket Prices')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()


'''
1.Single Variable Histograms:
    The first row of histograms shows the distribution of passengers by 'Sex', 'Pclass', and 'Survived'.
    We can see that there were more male passengers than female passengers.
    The majority of passengers were in the third class ('Pclass').
    The number of passengers who did not survive ('Survived=0') is higher than those who survived ('Survived=1').
2.Single Variable Histograms (Age and Fare):
    The second row of histograms shows the distribution of passenger ages and ticket fares.
    The age distribution appears to be somewhat right-skewed, with more passengers in the younger age groups.
    The fare distribution also appears to be right-skewed, indicating that most passengers paid lower fares.
3.Two Variable Histograms:
    The first set of histograms in this category shows the relationship between 'Age' and 'Fare', colored by 'Sex' and 'Survived'.
    It seems that there's no clear linear relationship between age and fare.
    The second set of histograms shows the relationship between 'Sex', 'Pclass', 'Age', and 'Fare', colored by 'Survived'.
    These visualizations help in understanding how survival might be related to different variables such as gender, class, age, and fare.
4.Correlation Heatmap:
    The correlation heatmap shows the correlations between numerical variables in the dataset.
    It appears that there is no strong correlation between variables, which suggests that they are mostly independent of each other.
5.Distribution of Ticket Prices (Fare):
    The histogram shows the distribution of ticket prices.
    As observed earlier, the distribution is right-skewed, indicating that most passengers paid lower fares.'''