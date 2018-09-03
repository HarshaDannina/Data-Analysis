import pandas as pd
import numpy as np
import quandl
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

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

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

#print(df.head())

x = np.array(df.drop(['label'],1))
y = np.array(df['label'])

x = preprocessing.scale(x)
y = np.array(df['label'])

#testing and training
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

#classifier
clf = LinearRegression(n_jobs=10)
#clf = svm.SVR()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print(accuracy)
