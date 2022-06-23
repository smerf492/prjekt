import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dane.csv', sep=",")
mar = pd.read_csv('marki.csv', sep=",")

cena=[10*i for i in range(30)]
for i in range(30):
    il=10*i
    if i>10: il=25*(i-6)
    cena[i]=il
ceil=[0 for i in range(30)]
lp=range(30)

for el in data['selling_price']:
    il=el/100000
    if il>10:
        il=el/250000+6
        if il>29: il=29
        #print(el)
    il=int(il)
    ceil[il]=ceil[il]+1
#print(ceil)

a=plt.bar(lp,ceil)
plt.bar_label(a,cena)
plt.show()

ma=mar['skrót']
lp=range(len(ma))
sr=data['selling_price'].mean()
wyk=[sr for i in range(len(ma))]
av=[0 for i in range(len(ma))]

for i in range(len(ma)):
    av[i]=data[data['name']==i]['selling_price'].mean()

a=plt.bar(lp,av)
plt.bar_label(a,ma)
plt.plot(lp,wyk,c='r')
plt.yscale('log')
plt.show()


moc=[10*i for i in range(20)]
for i in range(20):
    il=10*(i+2)
    if i>13: il=25*(i-7)
    if i>15: il=50*(i-11)
    moc[i]=il
moil=[0 for i in range(20)]
lp=range(20)

for el in data['max_power']:
    il=el/10-2
    if il>13: il=el/25+7
    if il>15: il=el/50+11
    if il>18: il=19
    il=int(il)
    moil[il]=moil[il]+1
#print(moil)

sr=data['selling_price'].mean()
wyk=[sr for i in range(len(moc))]

a=plt.bar(lp,moil)
plt.bar_label(a,moc)
plt.show()

av=[0 for i in range(len(moc))]

for i in range(len(moc)):
    if i<19: av[i]=data[(data['max_power']>=moc[i]) & (data['max_power']<moc[i+1])]['selling_price'].mean()
    if i==19: av[i]=data[data['max_power']>=moc[i]]['selling_price'].mean()

a=plt.bar(lp,av)
plt.bar_label(a,moc)
plt.plot(lp,wyk,c='r')
plt.yscale('log')
plt.show()


