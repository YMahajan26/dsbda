
# importing libaries
import pandas as pd
import numpy as np

df = pd.read_csv("2-Academic-Performance.csv")

# Display Null values
print('\n1. Total Number of Null Values in Dataset:\n', df.isna().sum())

# list of columns with na
print("\n2. List of name of columns with missing values")
cols_with_na = []
for col in df.columns:
    if df[col].isna().any():
        cols_with_na.append(col)
print("\n",cols_with_na,'\n')
# o/p 'Name', 'Phy_marks', 'Che_marks', 'EM1_marks', 'PPS_marks'

# we will drop rows with name na 
name_na = df[df['Name'].isna()]
print('\n3. Rows with name na : \n',name_na)
df = df.drop(name_na.index)
print('\n4. After dropping :',df.shape)

# we will fill other na with median but before than if we have any outliers replace them with na
# Mean replacement is often preferred when dealing with outliers because it's less sensitive to extreme values.
for col in cols_with_na:
    col_dt = df[col].dtypes
    if (col_dt == 'int64' or col_dt == 'float64'):
        outliers = (df[col] < 0) | (100 < df[col])
        df.loc[outliers, col] = np.nan
        df[col] = df[col].fillna(df[col].mean())  
print("\n",df.isna().sum())

df['Total Marks']=df['Phy_marks']+df['Che_marks']+df['EM1_marks']+df['PPS_marks']+df['SME_marks']
df['Percentage']=df['Total Marks']/5
print("\n",df.head())

# # Scan all numeric variables for outliers.# We'll use box plots to visualize outliers
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols=['Phy_marks','Che_marks','EM1_marks','PPS_marks','SME_marks','Percentage']
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[numeric_cols])
plt.title("Box plot of numeric variables")
plt.xticks()
plt.show()

# IQR method to detect outliers
print('\n5. Handling outliers:')
Q1 = df['Che_marks'].quantile(0.25)
Q3 = df['Che_marks'].quantile(0.75)
IQR = Q3 - Q1
Lower_limit = Q1 - 1.5 * IQR
Upper_limit = Q3 + 1.5 * IQR
print(f'\nQ1 = {Q1}, Q3 = {Q3}, IQR = {IQR}, Lower_limit = {Lower_limit}, Upper_limit = {Upper_limit}')
outliers = df[(df['Che_marks'] < Lower_limit) | (df['Che_marks'] > Upper_limit)]
print("\n",outliers) #Contextual Outliers

# step 3 handling outliers # remove outliers
df = df.drop(outliers.index)
print(df.shape)
# boxplot after removing outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[numeric_cols])
plt.title("Box plot of numeric variables")
plt.xticks()
plt.show()

#transformation on data on percentage (min max scaling) 
from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()
df['Percentage_Scaled'] = scaler.fit_transform(df[['Percentage']])
print('\n',df[['Percentage', 'Percentage_Scaled']])
print("\n.............................................................\n")
