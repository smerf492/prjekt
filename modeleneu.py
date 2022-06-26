import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint
from keras.callbacks import History
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


history = History()
history2 = History()
history3 = History()

#Ran=randint(1,1000)
Ran=248
print(Ran)
data = pd.read_csv('daneFULL.csv', sep=",")
X=data.drop(["selling_price"],axis=1)
y=data["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=Ran)
sc1=StandardScaler()
sc2=StandardScaler()
X_trains=sc1.fit_transform(X_train)
X_tests=sc1.transform(X_test)
wyn=[]
mod=[]
nazwy=[]
ep=5000
ba=10
early_stop = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)

def pomocnik(nazwa, grid1,train=X_train,test=X_test):
    wyn1=[nazwa]
    wyn1.append("OK")#grid1.best_params_)
    print("\n"+wyn1[0]+":", wyn1[1])
    wyn1.append(metrics.r2_score(y_train, grid1.predict(train)))
    wyn1.append(metrics.r2_score(y_test, grid1.predict(test)))
    wyn1.append(metrics.mean_absolute_percentage_error(y_train, grid1.predict(train)))
    wyn1.append(metrics.mean_absolute_percentage_error(y_test, grid1.predict(test)))
    print("r2/MAPE_train: {:1.10}".format(wyn1[2]),"  {:1.10}".format(wyn1[4]))
    print("r2/MAPE_test:  {:1.10}".format(wyn1[3]),"  {:1.10}".format(wyn1[5]))
    return wyn1

#Dense
nazwy.append("dense")
model = Sequential()
model.add(Dense(140, input_shape=(X_train.shape[1],)))
model.add(Activation("relu"))
model.add(Dense(70))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(1))
print(model.summary())
#model.compile(loss="mean_absolute_percentage_error", optimizer="Adam", metrics=["mean_squared_logarithmic_error"])
model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["mean_absolute_percentage_error"])
history = model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size=ba, epochs=ep, callbacks=[early_stop])
mod.append(model)


nazwy.append("dense standaryzowane dane")
model = Sequential()
model.add(Dense(140, input_shape=(X_trains.shape[1],)))
model.add(Activation("relu"))
model.add(Dense(70))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(1))
print(model.summary())
model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["mean_absolute_percentage_error"])
history2 = model.fit(X_trains, y_train, validation_data= (X_tests, y_test), batch_size=ba, epochs=ep, callbacks=[early_stop])
mod.append(model)


nazwy.append("dense standaryzowane dane + dropout")
model = Sequential()
model.add(Dense(140, input_shape=(X_trains.shape[1],)))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(70))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(1))
print(model.summary())
model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["mean_absolute_percentage_error"])
history3 = model.fit(X_trains, y_train, validation_data= (X_tests, y_test), batch_size=ba, epochs=ep, callbacks=[early_stop])
mod.append(model)


plt.plot(history.history['mean_absolute_percentage_error'], label='train')
plt.plot(history.history['val_mean_absolute_percentage_error'], label='test')
plt.plot(history2.history['mean_absolute_percentage_error'], label='st_train')
plt.plot(history2.history['val_mean_absolute_percentage_error'], label='st_test')
plt.plot(history3.history['mean_absolute_percentage_error'], label='st_dr_train')
plt.plot(history3.history['val_mean_absolute_percentage_error'], label='st_dr_test')
plt.grid(True)
plt.gca().set_ylim(0, 110)
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.plot(history2.history['loss'], label='st_train')
plt.plot(history2.history['val_loss'], label='st_test')
plt.plot(history3.history['loss'], label='st_dr_train')
plt.plot(history3.history['val_loss'], label='st_dr_test')
plt.grid(True)
plt.legend()
plt.show()

t0=mod[0].evaluate(X_test,y_test)
t1=mod[1].evaluate(X_tests,y_test)
t2=mod[2].evaluate(X_tests,y_test)
print()
print(nazwy[0],t0)
print(nazwy[1],t1)
print(nazwy[2],t2)

wyn.append(pomocnik(nazwy[0],mod[0]))
wyn.append(pomocnik(nazwy[1],mod[1],X_trains,X_tests))
wyn.append(pomocnik(nazwy[2],mod[2],X_trains,X_tests))

print()
print(mod[0].predict(X_test[0:3]))
print((mod[0].predict(X_test[0:3])).reshape(-1))
print((mod[1].predict(X_tests[0:3])).reshape(-1))
print((mod[2].predict(X_tests[0:3])).reshape(-1))
print(np.array(y_test[0:3]))

