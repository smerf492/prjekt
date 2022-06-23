import pandas as pd

data = pd.read_csv('dataset/Car details v3.csv', sep=",")
data.drop(['torque'],axis=1,inplace=True)

data['max_power']=data['max_power'].str.replace(' bhp','')
data['engine']=data['engine'].str.replace(' CC','')
data['mileage']=data['mileage'].str.replace(' kmpl','')
data['mileage']=data['mileage'].str.replace(' km/kg','')
data['name']=data['name'].str[0:4]

data.replace(['Petrol','Diesel','CNG','LPG'],[1,2,3,4],inplace=True)
data.replace(['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'],[1,2,3,4,0],inplace=True)
data.replace(['Manual','Automatic'],[0,1],inplace=True)
data.replace(['Individual','Dealer','Trustmark Dealer'],[1,2,3],inplace=True)

marki=['Amba','Asho','Audi','BMW ','Chev','Daew','Dats','Fiat','Forc','Ford','Hond','Hyun',
       'Isuz','Jagu','Jeep','Kia ','Land','Lexu','MG H','Mahi','Maru','Merc','Mits','Niss',
       'Opel','Peug','Rena','Skod','Tata','Toyo','Volk','Volv']
liczby=range(1,len(marki)+1)
data.replace(marki,liczby,inplace=True)

m = pd.DataFrame(marki, columns=["skrót"])
m.to_csv('marki.csv',sep=',',index=False)

data=data.dropna() #225/8128

print(data)
print(data.isna().sum())
#print(data.head(15))
#print(data['fuel'].unique())
#print(data['owner'].unique())
#print(data['transmission'].unique())
#print(data['seller_type'].unique())
print(data['year'].unique())
#print(sorted(data['name'].unique()))

data.to_csv('dane.csv',sep=',',index=False)
data = pd.read_csv('dane.csv', sep=",")
data=data.dropna()
data.to_csv('dane.csv',sep=',',index=False)

