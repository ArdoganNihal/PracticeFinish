import os
import pandas as pd
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')

df= pd.read_csv('C:\DataSet2\Traininig.csv')
df=df.drop(labels=['rn'], axis=1)
#print(df.head())
#print(df.shape)
y=df[['activity']]
x=df.iloc[:,1:]
#print(y)
x=x.sample(n=20, axis=1, random_state=1)
print(x.shape)
from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30, random_state=42)
