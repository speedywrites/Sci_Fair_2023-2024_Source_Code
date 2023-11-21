#all "print" statements are just benchmarks/tests
#all graphs open in a new link in ur browser
import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv('Real_Masterfile.csv' ,na_values='=')
#print(df)

#df.info()
#print(df.isnull().sum())
#print(df.head(6))
df.columns
data2=df.copy()
#replace null values with mean
data2=data2.fillna(data2.mean)
#print(data2.head())
#print(data2.info())

#mapping
dist=(data2['City'])
distset=set(dist)
dd=list(distset)
dictOfWords = { dd[i] : i for i in range(0, len(dd))}
data2['City']=data2['City'].map(dictOfWords)

dist=(data2['AQI_BUCKET'])
distset=set(dist)
dd=list(distset)
dictOfWords = { dd[i] : i for i in range(0, len(dd))}
data2['AQI_BUCKET']=data2['AQI_BUCKET'].map(dictOfWords)

data2["AQI_BUCKET"]=data2["AQI_BUCKET"].fillna(data2["AQI_BUCKET"].mean())

#print(data2)

#print(data2.isnull().sum())



#EDA(Analyse the data)


import plotly.express as px
fig1= px.scatter(df,x="PCPT",y="PCPT")
fig1.show()

import plotly.express as px
fig2= px.scatter(df,x="PCPT",y="MAX_T")
fig2.show()

import plotly.express as px
fig3= px.scatter(df,x="PCPT",y="MIN_T")
fig3.show()

import plotly.express as px
fig4= px.scatter(df,x="PCPT",y="AVG_T")
fig4.show()

import plotly.express as px
fig5= px.scatter(df,x="PCPT",y="SMOKE")
fig5.show()

import plotly.express as px
fig6= px.scatter(df,x="PCPT",y="PM10")
fig6.show()

import plotly.express as px
fig7= px.scatter(df,x="PCPT",y="DATE")
fig7.show()

import plotly.express as px
fig8= px.scatter(df,x="PCPT",y="AQI")
fig8.show()

import plotly.express as px
fig9= px.scatter(df,x="PCPT",y="SO2")
fig9.show()

import plotly.express as px
fig10= px.scatter(df,x="PCPT",y="O3")
fig10.show()

import plotly.express as px
fig11= px.scatter(df,x="MAX_T",y="PCPT")
fig11.show()

import plotly.express as px
fig12= px.scatter(df,x="MAX_T",y="MAX_T")
fig12.show()

import plotly.express as px
fig13= px.scatter(df,x="MAX_T",y="MIN_T")
fig13.show()

import plotly.express as px
fig14= px.scatter(df,x="MAX_T",y="AVG_T")
fig14.show()

import plotly.express as px
fig15= px.scatter(df,x="MAX_T",y="SMOKE")
fig15.show()

import plotly.express as px
fig16= px.scatter(df,x="MAX_T",y="PM10")
fig16.show()

import plotly.express as px
fig17= px.scatter(df,x="MAX_T",y="DATE")
fig17.show()

import plotly.express as px
fig18= px.scatter(df,x="MAX_T",y="AQI")
fig18.show()

import plotly.express as px
fig19= px.scatter(df,x="MAX_T",y="SO2")
fig19.show()

import plotly.express as px
fig20= px.scatter(df,x="MAX_T",y="O3")
fig20.show()

import plotly.express as px
fig21= px.scatter(df,x="MIN_T",y="PCPT")
fig21.show()

import plotly.express as px
fig22= px.scatter(df,x="MIN_T",y="MAX_T")
fig22.show()

import plotly.express as px
fig23= px.scatter(df,x="MIN_T",y="MIN_T")
fig23.show()

import plotly.express as px
fig24= px.scatter(df,x="MIN_T",y="AVG_T")
fig24.show()

import plotly.express as px
fig25= px.scatter(df,x="MIN_T",y="SMOKE")
fig25.show()

import plotly.express as px
fig26= px.scatter(df,x="MIN_T",y="PM10")
fig26.show()

import plotly.express as px
fig27= px.scatter(df,x="MIN_T",y="DATE")
fig27.show()

import plotly.express as px
fig28= px.scatter(df,x="MIN_T",y="AQI")
fig28.show()

import plotly.express as px
fig29= px.scatter(df,x="MIN_T",y="SO2")
fig29.show()

import plotly.express as px
fig30= px.scatter(df,x="MIN_T",y="O3")
fig30.show()

import plotly.express as px
fig31= px.scatter(df,x="AVG_T",y="PCPT")
fig31.show()

import plotly.express as px
fig32= px.scatter(df,x="AVG_T",y="MAX_T")
fig32.show()

import plotly.express as px
fig33= px.scatter(df,x="AVG_T",y="MIN_T")
fig33.show()

import plotly.express as px
fig34= px.scatter(df,x="AVG_T",y="AVG_T")
fig34.show()

import plotly.express as px
fig35= px.scatter(df,x="AVG_T",y="SMOKE")
fig35.show()

import plotly.express as px
fig36= px.scatter(df,x="AVG_T",y="PM10")
fig36.show()

import plotly.express as px
fig37= px.scatter(df,x="AVG_T",y="DATE")
fig37.show()

import plotly.express as px
fig38= px.scatter(df,x="AVG_T",y="AQI")
fig38.show()

import plotly.express as px
fig39= px.scatter(df,x="AVG_T",y="SO2")
fig39.show()

import plotly.express as px
fig40= px.scatter(df,x="AVG_T",y="O3")
fig40.show()

import plotly.express as px
fig41= px.scatter(df,x="SMOKE",y="PCPT")
fig41.show()

import plotly.express as px
fig42= px.scatter(df,x="SMOKE",y="MAX_T")
fig42.show()

import plotly.express as px
fig43= px.scatter(df,x="SMOKE",y="MIN_T")
fig43.show()

import plotly.express as px
fig44= px.scatter(df,x="SMOKE",y="AVG_T")
fig44.show()

import plotly.express as px
fig45= px.scatter(df,x="SMOKE",y="SMOKE")
fig45.show()

import plotly.express as px
fig46= px.scatter(df,x="SMOKE",y="PM10")
fig46.show()

import plotly.express as px
fig47= px.scatter(df,x="SMOKE",y="DATE")
fig47.show()

import plotly.express as px
fig48= px.scatter(df,x="SMOKE",y="AQI")
fig48.show()

import plotly.express as px
fig49= px.scatter(df,x="SMOKE",y="SO2")
fig49.show()

import plotly.express as px
fig50= px.scatter(df,x="SMOKE",y="O3")
fig50.show()

import plotly.express as px
fig51= px.scatter(df,x="PM10",y="PCPT")
fig51.show()

import plotly.express as px
fig52= px.scatter(df,x="PM10",y="MAX_T")
fig52.show()

import plotly.express as px
fig53= px.scatter(df,x="PM10",y="MIN_T")
fig53.show()

import plotly.express as px
fig54= px.scatter(df,x="PM10",y="AVG_T")
fig54.show()

import plotly.express as px
fig55= px.scatter(df,x="PM10",y="SMOKE")
fig55.show()

import plotly.express as px
fig56= px.scatter(df,x="PM10",y="PM10")
fig56.show()

import plotly.express as px
fig57= px.scatter(df,x="PM10",y="DATE")
fig57.show()

import plotly.express as px
fig58= px.scatter(df,x="PM10",y="AQI")
fig58.show()

import plotly.express as px
fig59= px.scatter(df,x="PM10",y="SO2")
fig59.show()

import plotly.express as px
fig60= px.scatter(df,x="PM10",y="O3")
fig60.show()

import plotly.express as px
fig61= px.scatter(df,x="DATE",y="PCPT")
fig61.show()

import plotly.express as px
fig62= px.scatter(df,x="DATE",y="MAX_T")
fig62.show()

import plotly.express as px
fig63= px.scatter(df,x="DATE",y="MIN_T")
fig63.show()

import plotly.express as px
fig64= px.scatter(df,x="DATE",y="AVG_T")
fig64.show()

import plotly.express as px
fig65= px.scatter(df,x="DATE",y="SMOKE")
fig65.show()

import plotly.express as px
fig66= px.scatter(df,x="DATE",y="PM10")
fig66.show()

import plotly.express as px
fig67= px.scatter(df,x="DATE",y="DATE")
fig67.show()

import plotly.express as px
fig68= px.scatter(df,x="DATE",y="AQI")
fig68.show()

import plotly.express as px
fig69= px.scatter(df,x="DATE",y="SO2")
fig69.show()

import plotly.express as px
fig70= px.scatter(df,x="DATE",y="O3")
fig70.show()

import plotly.express as px
fig71= px.scatter(df,x="AQI",y="PCPT")
fig71.show()

import plotly.express as px
fig72= px.scatter(df,x="AQI",y="MAX_T")
fig72.show()

import plotly.express as px
fig73= px.scatter(df,x="AQI",y="MIN_T")
fig73.show()

import plotly.express as px
fig74= px.scatter(df,x="AQI",y="AVG_T")
fig74.show()

import plotly.express as px
fig75= px.scatter(df,x="AQI",y="SMOKE")
fig75.show()

import plotly.express as px
fig76= px.scatter(df,x="AQI",y="PM10")
fig76.show()

import plotly.express as px
fig77= px.scatter(df,x="AQI",y="DATE")
fig77.show()

import plotly.express as px
fig78= px.scatter(df,x="AQI",y="AQI")
fig78.show()

import plotly.express as px
fig79= px.scatter(df,x="AQI",y="SO2")
fig79.show()

import plotly.express as px
fig80= px.scatter(df,x="AQI",y="O3")
fig80.show()

import plotly.express as px
fig81= px.scatter(df,x="SO2",y="PCPT")
fig81.show()

import plotly.express as px
fig82= px.scatter(df,x="SO2",y="MAX_T")
fig82.show()

import plotly.express as px
fig83= px.scatter(df,x="SO2",y="MIN_T")
fig83.show()

import plotly.express as px
fig84= px.scatter(df,x="SO2",y="AVG_T")
fig84.show()

import plotly.express as px
fig85= px.scatter(df,x="SO2",y="SMOKE")
fig85.show()

import plotly.express as px
fig86= px.scatter(df,x="SO2",y="PM10")
fig86.show()

import plotly.express as px
fig87= px.scatter(df,x="SO2",y="DATE")
fig87.show()

import plotly.express as px
fig88= px.scatter(df,x="SO2",y="AQI")
fig88.show()

import plotly.express as px
fig89= px.scatter(df,x="SO2",y="SO2")
fig89.show()

import plotly.express as px
fig90= px.scatter(df,x="SO2",y="O3")
fig90.show()

import plotly.express as px
fig91= px.scatter(df,x="O3",y="PCPT")
fig91.show()

import plotly.express as px
fig92= px.scatter(df,x="O3",y="MAX_T")
fig92.show()

import plotly.express as px
fig93= px.scatter(df,x="O3",y="MIN_T")
fig93.show()

import plotly.express as px
fig94= px.scatter(df,x="O3",y="AVG_T")
fig94.show()

import plotly.express as px
fig95= px.scatter(df,x="O3",y="SMOKE")
fig95.show()

import plotly.express as px
fig96= px.scatter(df,x="O3",y="PM10")
fig96.show()

import plotly.express as px
fig97= px.scatter(df,x="O3",y="DATE")
fig97.show()

import plotly.express as px
fig98= px.scatter(df,x="O3",y="AQI")
fig98.show()

import plotly.express as px
fig99= px.scatter(df,x="O3",y="SO2")
fig99.show()

import plotly.express as px
fig100= px.scatter(df,x="O3",y="O3")
fig100.show()



#Everything below this is predicting the future AQI

#preprocessing, feature selection
#train and test split data
#training multiple models

data2.info
data2.columns

features = data2[['O3', 'SO2', 'Precipitation', 'Average_Temperature', 'Smoke']]

labels = data2['AQI']

#splitting into test and train data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,labels,test_size = 0.2, random_state =2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import r2_score 


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
#X, Y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(Xtrain, Ytrain)
print(regr.predict(Xtest))

y_pred=regr.predict(Xtest)
from sklearn.metrics import r2_score
print(r2_score(Ytest, y_pred))