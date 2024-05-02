
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

tdata=pd.read_csv("9-titanic.csv")

print(tdata.head(10))
print(".............................................................................................................")

print(tdata.info())
print(".............................................................................................................")

print(tdata.isnull().sum())
print(".............................................................................................................")

# assigning values ot null
# tdata['Age'] = tdata['Age'].fillna(np.mean(tdata['Age']))

tdata['Age'] = tdata['Age'].fillna(tdata['Age'].mean())
tdata['Cabin'] = tdata['Cabin'].fillna(tdata['Cabin'].mode()[0])
tdata['Embarked'] = tdata['Embarked'].fillna(tdata['Embarked'].mode()[0])

#checking if we removed all the missing values
print(tdata.isnull().sum())
print(".....................................................................................................................")

# boxplot
plt.figure(figsize=(10, 9))
sns.boxplot(x='Sex', y='Age', hue='Survived', data=tdata)
plt.title('Distribution of Age by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')

plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()





