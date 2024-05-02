
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# FOR Employee_Salary.csv
df = pd.read_csv('3-Employee_Salary.csv')
# Display Null values
print('\nTotal Number of Null Values in Dataset:\n', df.isna().sum())

# Fill the missing values
# df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df.fillna({'Gender':df['Gender'].mode()[0]}, inplace=True)

# df['Experience'].fillna(df['Experience'].mean(), inplace=True)
df.fillna({'Experience':df['Experience'].mean()}, inplace=True)

print('\nTotal Number of Null Values in Dataset:\n', df.isna().sum())

# Display Overall Statistical information
print('\nStatistical information of Numerical Columns: \n',df.describe())

# groupwise statistical information
print('\nGroupwise Statistical Summary....')
print('\n-------------------------- Experience -----------------------\n')
print(df['Experience'].groupby(df['Gender']).describe())
print('\n-------------------------- Age -----------------------\n')
print(df['Age'].groupby(df['Gender']).describe())
print('\n-------------------------- Salary -----------------------\n')
print(df['Salary'].groupby(df['Gender']).describe())


#---------------------------------------------------------------------------------------
# FOR Iris.csv
#---------------------------------------------------------------------------------------
# Read the data from CSV file
data = pd.read_csv("3-Iris.csv")
# Filter the data for the desired species
setosa_data = data[data['Species'] == 'Iris-setosa']
versicolor_data = data[data['Species'] == 'Iris-versicolor']
virginica_data = data[data['Species'] == 'Iris-virginica']

# Display basic statistical details for each species
print("Summary statistics for Iris-setosa:")
print(setosa_data.describe())

print("\nSummary statistics for Iris-versicolor:")
print(versicolor_data.describe())

print("\nSummary statistics for Iris-virginica:")
print(virginica_data.describe())

# Create boxplots for each feature
sns.boxplot(data=data, x='Species', y='SepalLengthCm')
plt.show()
sns.boxplot(data=data, x='Species', y='SepalWidthCm')
plt.show()
sns.boxplot(data=data, x='Species', y='PetalLengthCm')
plt.show()
sns.boxplot(data=data, x='Species', y='PetalWidthCm')
plt.show()