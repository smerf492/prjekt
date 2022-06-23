import pandas as pd

data = pd.read_csv('daneFULL.csv', sep=",")

data2 = data.head(1200)

data.to_csv('daneFULL.csv',sep=',',index=False)
data2.to_csv('dane.csv',sep=',',index=False)