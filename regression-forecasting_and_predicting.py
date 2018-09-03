import pandas as pd
import numpy as np
import quandl
import math
import datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = ''

df = quandl.get("WIKI/GOOGL")

#print(df.head())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0

#Daily percent change
df['PT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] *100.0

#New Data Frame
df = df[['Adj. Close', 'HL_PCT', 'PT_change', 'Adj. Volume']]

#forecast the value
forecast_col = 'Adj. Close'
df.fillna(value = -99999, inplace = True)

forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.head())

x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out:]

df.dropna(inplace=True)

y = np.array(df['label'])
y = np.array(df['label'])


#testing and training
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

#classifier
clf = LinearRegression(n_jobs=-1)
#clf = svm.SVR()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print(accuracy)

forecast_set = clf.predict(x_lately)

#print(forecast_set)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

#print(df.head())
print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
