import numpy as np
import pandas as pd
from random import randint
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#Ran=randint(1,1000)
Ran=248
print(Ran)
data = pd.read_csv('daneFULL.csv', sep=",")
X=data.drop(["selling_price"],axis=1)
y=data["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=Ran)
kfold = KFold(n_splits=5)
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


#RBF
nazwy.append("svc_rbf")
pipe = Pipeline([('pre', StandardScaler()), ('reg',svm.SVC())])


param_grid = {'pre': [StandardScaler()],
              'reg__C': [100, 1000, 10000, 100000, 1000000],
              'reg__gamma': [0.1, 1, 10, 100, 1000]}

grid.append(GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))


param_grid2 = {'pre': [None],
              'reg__C': [10, 100, 1000, 10000, 100000],
              'reg__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid.append(GridSearchCV(pipe, param_grid2, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))

print("OK")
#poly
nazwy.append("svc_poly")
pipe3 = Pipeline([('pre', StandardScaler()), ('reg',svm.SVC(kernel='poly'))])


param_grid3 = {'pre': [StandardScaler()], #None zajmuje za dużo czasu
              'reg__C': [0.1, 1, 10, 100, 1000],
              'reg__degree': [1, 2, 3],
              'reg__coef0': [0, 0.5, 1],
              'reg__gamma': [0.1, 1, 10, 100, 1000]}

grid.append(GridSearchCV(pipe3, param_grid3, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))

from joblib import dump
i=0
while i<len(wyn):
    dump(grid[i],wyn[i][0]+str(i+10)+'.model')
    i=i+1


