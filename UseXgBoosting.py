import os
import pandas as pd
import numpy as np

from warnings import filterwarnings

from sklearn.metrics import accuracy_score

filterwarnings('ignore')

df= pd.read_csv('C:\DataSet2\Traininig.csv')
df=df.drop(labels=['rn'], axis=1)
#print(df.head())
#print(df.shape)
y=df[['activity']]
x=df.iloc[:,1:]
#print(y)
x=x.sample(n=20, axis=1, random_state=1)
#print(x.shape)
from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30, random_state=42)

#Gradient Boosting Uygulaması
from sklearn.ensemble import  GradientBoostingClassifier
from  time import time
t0=time()
gbm_model=GradientBoostingClassifier()
gbm_model.fit(x_train, y_train)
gbm_time= time() - t0;
gbm_acc=accuracy_score(y_test, gbm_model.predict(x_test))

#XGBoost Ugulamsı

from xgboost import XGBClassifier
t0=time()
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)
XGBoost_time= time()-t0
XGBoost_acc = accuracy_score(y_test, xgb_model.predict(x_test))
#print(XGBoost_acc)

#LightGBM Uygulaması
import lightgbm as lgb
t0=time()
lgb_model=lgb.LGBMClassifier()
lgb_model.fit(x_train, y_train)
lgb_time=time()- t0
lgb_acc = accuracy_score(y_test, lgb_model.predict(x_test))

#Görselleştirme
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()
object=['GBM', 'XGBoost', 'LightGBM']
y_pos=np.arange(len(object))
performans_time =[gbm_time, XGBoost_time,lgb_time ]
performan_acc=[gbm_acc, XGBoost_acc, lgb_acc]
ax1.set_ylabel('saniye')
ax1.bar(y_pos, performans_time, color='pink')
ax1.tick_params(axis='y')
ax2=ax1.twinx()
ax2.set_ylabel('Doğru Tahmin Oranı')
ax2.plot(y_pos, performan_acc, color='red')
plt.xticks(y_pos, object)
fig.tight_layout()
plt.title('Model Performansları')
plt.show()
