from operator import le
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler



dataframe = pd.read_csv('C:\Users\Behrad Edraki\Desktop\\traffic.csv')


dataframe["DateTime"]= pd.to_datetime(dataframe["DateTime"])
dataframe = dataframe.drop(['ID'],axis=1)
dataframe["Year"]= dataframe['DateTime'].dt.year
dataframe["Month"]= dataframe['DateTime'].dt.month
dataframe["Date_no"]= dataframe['DateTime'].dt.day
dataframe["Hour"]= dataframe['DateTime'].dt.hour
dataframe["Day"]= dataframe.DateTime.dt.strftime("%A")

print(dataframe.head(20))



X = dataframe[['Year','Month','Date_no','Hour','Junction']]
y = dataframe['Vehicles']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)





#sns.lineplot(x=dataframe['DateTime'],y="Vehicles",data=dataframe, hue="Junction", )
#plt.show()

regr = SVR(kernel='rbf')
regr.fit(x_train, y_train)
#regr.fit(dataframe[['Year','Month','Date_no','Hour','Junction']], dataframe['Vehicles'])# >>>>>>>>>>>> it's worked <<<<<<<<<<<

# Generate predictions for testing data
#pred = regr.predict([['2018','11','2','0','2']])# >>>>>>>>>>>> it's worked, But!!! I need to split dataframe to train and test <<<<<<<<<<<
pred = regr.predict(x_test)


df1=(pred-y_test)/y_test
df1=round(df1.mean()*100,2)
print("Error = ",df1,"%") 
a=100-df1
print("Accuracy= ",a,"%")
