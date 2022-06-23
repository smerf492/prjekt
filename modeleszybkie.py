import numpy as np
import pandas as pd
from random import randint
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
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


#LinearRegression
nazwy.append("linreg")
pipe = Pipeline([('pre', StandardScaler()), ('poly',PolynomialFeatures(degree=2)), ('lire',lm.LinearRegression())])


param_grid = {'pre': [StandardScaler()],
              'poly__degree': [1, 2, 3, 4]}

grid.append(GridSearchCV(pipe, param_grid, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))


param_grid2 = {'pre': [None],
              'poly__degree': [1, 2, 3, 4]}

grid.append(GridSearchCV(pipe, param_grid2, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))


#LassoRegression
nazwy.append("lasso")
pipe3 = Pipeline([('pre', StandardScaler()), ('poly',PolynomialFeatures()), ('lire',lm.Lasso(tol=0.1))])


param_grid3 = {'pre': [StandardScaler()],
              'poly__degree': [1, 2, 3, 4, 5],
              'lire__alpha': [1, 10, 100, 1000, 10000, 100000]}

grid.append(GridSearchCV(pipe3, param_grid3, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))


param_grid4 = {'pre': [None],
              'poly__degree': [3, 4, 5, 6, 7],
              'lire__alpha': [1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02]}

grid.append(GridSearchCV(pipe3, param_grid4, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))


#RidgeRegression
nazwy.append("ridge")
pipe5 = Pipeline([('pre', StandardScaler()), ('poly',PolynomialFeatures()), ('lire',lm.Ridge())])


param_grid5 = {'pre': [StandardScaler()],
              'poly__degree': [1, 2, 3, 4],
              'lire__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid.append(GridSearchCV(pipe5, param_grid5, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))


param_grid6 = {'pre': [None],
              'poly__degree': [1, 2, 3, 4],
              'lire__alpha': [1e-05, 1e-03, 1e-02, 0.1, 1, 10]}

grid.append(GridSearchCV(pipe5, param_grid6, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))


#RandomForestRegressor
nazwy.append("ran_forest")
pipe7 = Pipeline([('pre', StandardScaler()), ('fre',rfr())])


param_grid7 = {'pre': [StandardScaler()],
              'fre__max_depth': [10, 15, 20, 25],
              'fre__n_estimators': [10, 15, 20, 25, 30],
              'fre__random_state': [1, 2, 3, 4, 5]}

grid.append(GridSearchCV(pipe7, param_grid7, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))


param_grid8 = {'pre': [None],
              'fre__max_depth': [10, 15, 20, 25],
              'fre__n_estimators': [10, 15, 20, 25, 30],
              'fre__random_state': [1, 2, 3, 4, 5]}

grid.append(GridSearchCV(pipe7, param_grid8, cv=kfold, return_train_score=True))
grid[-1].fit(X_train, y_train)
wyn.append(pomocnik(nazwy[-1], grid[-1]))


print(Ran)
for line in wyn:
   print()
   print(line[0:2])
   print(line[2:])

from joblib import dump
i=0
while i<len(wyn):
    dump(grid[i],wyn[i][0]+str(i)+'.model')
    i=i+1


#print(wyn)