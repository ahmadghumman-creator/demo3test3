import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import warnings
warnings.filterwarnings('ignore')

titanic_df = pd.read_csv('data_processed.csv')

def titanic_children(passenger):
    age , sex = passenger
    if age <16:
        return 'child'
    return sex

titanic_df['person'] = titanic_df[['Age','Sex']].apply(titanic_children,axis=1)

titanic_df["Title"] = titanic_df.Name.str.extract("([A-Za-z]+)\.")

titanic_df['has_cabin']=1

titanic_df.loc[titanic_df.Cabin == "UKN",'has_cabin'] = 0

titanic_df.drop(['Name','Cabin'],inplace=True,axis=1)

titanic_df['Family_Members'] = titanic_df['SibSp'] + titanic_df['Parch']

for Col in ['Sex','Embarked','person','Title']:
    Unique_entries = titanic_df[Col].unique()
    D = {}
    for i in range(len(Unique_entries)):
        D[Unique_entries[i]]=i
    print(D)
    
for Col in ['Sex','Embarked','person','Title']:
    Unique_entries = titanic_df[Col].unique()
    D = {}
    for i in range(len(Unique_entries)):
        D[Unique_entries[i]]=i
    titanic_df[Col] = titanic_df[Col].map(D)
    
titanic_df.to_csv('data_features.csv',index=False)