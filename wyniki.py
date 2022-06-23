import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn import metrics
from sklearn.model_selection import train_test_split


Ran=248
data = pd.read_csv('daneFULL.csv', sep=",")
X=data.drop(["selling_price"],axis=1)
y=data["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=Ran)
wyn=[]
grid=[]
nazwy=[]

def pomocnik(nazwa, grid1):
    wyn1=[nazwa]
    wyn1.append(grid1.best_params_)
    print("\n"+wyn1[0]+":", wyn1[1])
    wyn1.append(metrics.r2_score(y_train, grid1.predict(X_train)))
    wyn1.append(metrics.r2_score(y_test, grid1.predict(X_test)))
    wyn1.append(metrics.mean_absolute_percentage_error(y_train, grid1.predict(X_train)))
    wyn1.append(metrics.mean_absolute_percentage_error(y_test, grid1.predict(X_test)))
    print("r2/MAPE_train: {:1.10}".format(wyn1[2]),"  {:1.10}".format(wyn1[4]))
    print("r2/MAPE_test:  {:1.10}".format(wyn1[3]),"  {:1.10}".format(wyn1[5]))
    return wyn1

nazwy.append("linreg")
nazwy.append("linreg")
nazwy.append("lasso")
nazwy.append("lasso")
nazwy.append("ridge")
nazwy.append("ridge")
nazwy.append("ran_forest")
nazwy.append("ran_forest")
nazwy.append("svc_rbf")
nazwy.append("svc_rbf")
nazwy.append("svc_poly")


from joblib import load
i=0
print(Ran)
while i<len(nazwy):
    grid.append(load(nazwy[i]+str(i)+'.model'))
    wyn.append(pomocnik(nazwy[i],grid[i]))
    i=i+1

'''
print(Ran)
for line in wyn:
   print()
   print(line[0:2])
   print(line[2:])
'''

