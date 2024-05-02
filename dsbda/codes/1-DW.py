import pandas as pd

df = pd.read_csv("1-iris.csv")

# Display basic information
print('\n1. Information of Dataset:\n', df.info)
print('\n2. Shape of Dataset (row x column):', df.shape)
print('\n3. Columns Name: ', df.columns)
print('\n4. Total elements in dataset: ', df.size)
print("\n5. Index: ", df.index)
print('\n6. First 5 rows:\n', df.head())
print('\n7. Last 5 rows:\n',df.tail())
print('\n8. Any 5 rows:\n',df.sample(5))
print("\n9. Datatype of attributes (columns):\n", df.dtypes)

# Display Statistical information
print('\n10. Statistical information of Numerical Columns: \n',df.describe())

# changing dtype of 'Species' column
df['Species'] = df['Species'].astype("string")
print("\n11. dtypes(after changing 'Species' datatype):\n", df.dtypes)

# print("\n12. Displaying rows from 15-20:\n", df[15:20])
# print("\n13. Displaying row at location 15:\n", df.iloc[15])
# print("\n14. loc[15:20,['SepalLength','SepalWidth']:\n", df.loc[15:20,['SepalLength','SepalWidth']])
# cols_2_4 = df.columns[3:5]
# print("\n15. Displaying columns from 2 to 4:\n", df[cols_2_4])

# Display Null values
print('16. Displaying Null values in Dataset:')
df1 = df
print("\ni] isnull().any():\n", df1.isnull().any())
print("\nii] isna().any():\n", df1.isna().any())
print("\niii] isnull().sum():\n", df1.isnull().sum())
missing_values = df1[df1['SepalLength'].isnull() | df1['PetalWidth'].isnull() ]
print("\niv] Displaying rows with missing values: \n", missing_values)
df2 = df1.drop(missing_values.index)
print("\nv] Shape after dropping missing values: \n", df2.shape)


# Normalization of data
# converting the range of data into uniform range
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x=df2.iloc[:,1:5]
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled)
print("\n17. Normalized dataset using MinMaxScaler:\n", df_normalized)

# Converting categorical (qualitative) variable to numeric (quantitative) variable
print('\n18. Handling categorical variables: \n')
# 1->(Label Encoding)
print("[A] Label Encoding: ")
from sklearn import preprocessing
print("\ni] unique values in species before encoding:", df['Species'].unique())

label_encoder = preprocessing.LabelEncoder()
df['Species']= label_encoder.fit_transform(df['Species'])
print("\nii] unique values in species after encoding:\n",df['Species'].unique())
print("\n=========\n",df.head(5))

# encodeddata = pd.get_dummies(irisdata, columns=['Species'])
# print(encodeddata.head(5))

# """ OUTPUT: 
# 1. Information of Dataset:
#  <bound method DataFrame.info of       Id  SepalLength  SepalWidth  PetalLength  PetalWidth    Species
# 0      1          5.1         3.5          1.4         0.2     Setosa
# 1      2          4.9         3.0          1.4         0.2     Setosa
# 2      3          4.7         3.2          1.3         0.2     Setosa
# 3      4          4.6         3.1          1.5         0.2     Setosa
# 4      5          5.0         3.6          1.4         0.2     Setosa
# ..   ...          ...         ...          ...         ...        ...
# 145  146          6.7         3.0          5.2         2.3  Virginica
# 146  147          6.3         2.5          5.0         1.9  Virginica
# 147  148          6.5         3.0          5.2         2.0  Virginica
# 148  149          6.2         3.4          5.4         2.3  Virginica
# 149  150          5.9         3.0          5.1         1.8  Virginica

# [150 rows x 6 columns]>

# 2. Shape of Dataset (row x column): (150, 6)

# 3. Columns Name:  Index(['Id', 'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth',
#        'Species'],
#       dtype='object')

# 4. Total elements in dataset:  900

# 5. Index:  RangeIndex(start=0, stop=150, step=1)

# 6. First 5 rows:
#     Id  SepalLength  SepalWidth  PetalLength  PetalWidth Species
# 0   1          5.1         3.5          1.4         0.2  Setosa
# 1   2          4.9         3.0          1.4         0.2  Setosa
# 2   3          4.7         3.2          1.3         0.2  Setosa
# 3   4          4.6         3.1          1.5         0.2  Setosa
# 4   5          5.0         3.6          1.4         0.2  Setosa

# 7. Last 5 rows:
#        Id  SepalLength  SepalWidth  PetalLength  PetalWidth    Species
# 145  146          6.7         3.0          5.2         2.3  Virginica
# 146  147          6.3         2.5          5.0         1.9  Virginica
# 147  148          6.5         3.0          5.2         2.0  Virginica
# 148  149          6.2         3.4          5.4         2.3  Virginica
# 149  150          5.9         3.0          5.1         1.8  Virginica

# 8. Any 5 rows:
#        Id  SepalLength  SepalWidth  PetalLength  PetalWidth    Species
# 5      6          5.4         3.9          1.7         0.4     Setosa
# 19    20          5.1         3.8          1.5         NaN     Setosa
# 27    28          5.2         3.5          1.5         0.2     Setosa
# 14    15          5.8         4.0          1.2         0.2     Setosa
# 111  112          6.4         2.7          5.3         1.9  Virginica

# 9. Datatype of attributes (columns):
#  Id               int64
# SepalLength    float64
# SepalWidth     float64
# PetalLength    float64
# PetalWidth     float64
# Species         object
# dtype: object

# 10. Statistical information of Numerical Columns:
#                 Id  SepalLength  SepalWidth  PetalLength  PetalWidth
# count  150.000000   148.000000  150.000000   150.000000  148.000000
# mean    75.500000     5.856757    3.054000     3.758667    1.197973
# std     43.445368     0.825459    0.433594     1.764420    0.760278
# min      1.000000     4.300000    2.000000     1.000000    0.100000
# 25%     38.250000     5.100000    2.800000     1.600000    0.300000
# 50%     75.500000     5.800000    3.000000     4.350000    1.300000
# 75%    112.750000     6.400000    3.300000     5.100000    1.800000
# max    150.000000     7.900000    4.400000     6.900000    2.500000

# 11. dtypes(after changing 'Species' datatype):
#  Id                      int64
# SepalLength           float64
# SepalWidth            float64
# PetalLength           float64
# PetalWidth            float64
# Species        string[python]
# dtype: object

# 12. Displaying rows from 15-20:
#      Id  SepalLength  SepalWidth  PetalLength  PetalWidth Species
# 15  16          5.7         4.4          1.5         0.4  Setosa
# 16  17          5.4         3.9          1.3         0.4  Setosa
# 17  18          5.1         3.5          1.4         0.3  Setosa
# 18  19          5.7         3.8          1.7         0.3  Setosa
# 19  20          5.1         3.8          1.5         NaN  Setosa

# 13. Displaying row at location 15:
#  Id                 16
# SepalLength       5.7
# SepalWidth        4.4
# PetalLength       1.5
# PetalWidth        0.4
# Species        Setosa
# Name: 15, dtype: object

# 14. loc[15:20,['SepalLength','SepalWidth']:
#      SepalLength  SepalWidth
# 15          5.7         4.4
# 16          5.4         3.9
# 17          5.1         3.5
# 18          5.7         3.8
# 19          5.1         3.8
# 20          5.4         3.4

# 15. Displaying columns from 2 to 4:
#       PetalLength  PetalWidth
# 0            1.4         0.2
# 1            1.4         0.2
# 2            1.3         0.2
# 3            1.5         0.2
# 4            1.4         0.2
# ..           ...         ...
# 145          5.2         2.3
# 146          5.0         1.9
# 147          5.2         2.0
# 148          5.4         2.3
# 149          5.1         1.8

# [150 rows x 2 columns]
# 16. Displaying Null values in Dataset:

# i] isnull().any():
#  Id             False
# SepalLength     True
# SepalWidth     False
# PetalLength    False
# PetalWidth      True
# Species        False
# dtype: bool

# ii] isna().any():
#  Id             False
# SepalLength     True
# SepalWidth     False
# PetalLength    False
# PetalWidth      True
# Species        False
# dtype: bool

# iii] isnull().sum():
#  Id             0
# SepalLength    2
# SepalWidth     0
# PetalLength    0
# PetalWidth     2
# Species        0
# dtype: int64

# iv] Displaying rows with missing values:
#        Id  SepalLength  SepalWidth  PetalLength  PetalWidth    Species
# 12    13          NaN         3.0          1.4         0.1     Setosa
# 19    20          5.1         3.8          1.5         NaN     Setosa
# 34    35          NaN         3.1          1.5         0.1     Setosa
# 104  105          6.5         3.0          5.8         NaN  Virginica

# v] Shape after dropping missing values:
#  (146, 6)

# 17. Normalized dataset using MinMaxScaler:
#              0         1         2         3
# 0    0.222222  0.625000  0.067797  0.041667
# 1    0.166667  0.416667  0.067797  0.041667
# 2    0.111111  0.500000  0.050847  0.041667
# 3    0.083333  0.458333  0.084746  0.041667
# 4    0.194444  0.666667  0.067797  0.041667
# ..        ...       ...       ...       ...
# 141  0.666667  0.416667  0.711864  0.916667
# 142  0.555556  0.208333  0.677966  0.750000
# 143  0.611111  0.416667  0.711864  0.791667
# 144  0.527778  0.583333  0.745763  0.916667
# 145  0.444444  0.416667  0.694915  0.708333

# [146 rows x 4 columns]

# 18. Handling categorical variables:

# [A] Label Encoding:

# i] unique values in species before encoding: <StringArray>
# ['Setosa', 'Versicolor', 'Virginica']
# Length: 3, dtype: string

# ii] unique values in species after encoding:
#  [0 1 2]

# """