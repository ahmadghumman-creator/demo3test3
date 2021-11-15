import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')

titanic_df = pd.read_csv('data/data_raw.csv')

print(titanic_df.head())

print(titanic_df.columns.values)

print(titanic_df.dtypes)

print(titanic_df.shape)

a = sns.catplot('Pclass',data=titanic_df,hue='Sex',kind='count')
a.savefig('figure.png')

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())

titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')

titanic_df.Cabin.fillna(value="UKN", inplace=True)

titanic_df.drop(['PassengerId', 'Ticket' ], axis=1, inplace=True)

titanic_df.to_csv('data_processed.csv',index=False)