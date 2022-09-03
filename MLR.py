import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#import dataset
startup_df=pd.read_csv(r'50_Startups.csv')
startup_df
shape=startup_df.shape
print("Dataset contains {} rows and {} columns".format(shape[0],shape[1]))
startup_df.columns
#Statistical Details of the dataset
startup_df.describe()
x=startup_df.iloc[:,:4]
y=startup_df.iloc[:,4]
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
x=ohe.fit_transform(startup_df[['State']])
x
ohe.categories_
from sklearn.compose import make_column_transformer
col_trans=make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'),['State']),
    remainder='passthrough')
x=col_trans.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#shapes of splitted data
print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)
