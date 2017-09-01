import numpy as np
import pandas as pd
import math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

#-->These Code Block is for refrence only<--#
#	this was used to prepare the original data(foramatting dates,
#                                              cutting some columns...)
#	we use the prepared data--> "IRKH-daily.csv"

# df = pd.read_csv("IRKH-ticker-daily.csv",
# 	usecols=['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'])
# # dates were int, converting to 'hyphen' dates, YYYYMMDD
# df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')

# # writing to a csv file so others may not do this procedure again!
# df.to_csv("IRKH-daily.csv")

#-->End of Block<--#

# define date parser for reading dates as dates in dataframe
dateparser = lambda d: pd.datetime.strptime(d, '%Y-%m-%d')
#load data with 'DATE' as index
df = pd.read_csv("IRKH-daily.csv", index_col='DATE', 
									parse_dates=['DATE'], 
								    date_parser=dateparser)

# save dates and closing price for easy future refrences
dates = df.index.values
close_price = df['CLOSE']

# generate some 'meaningful' feature set for training
df['HL_PCT'] = (df['HIGH'] - df['LOW']) / df['HIGH'] * 100.0
df['OC_PCT'] = (df['OPEN'] - df['CLOSE']) / df['OPEN'] * 100.0
df = df[['CLOSE', 'HL_PCT', 'OC_PCT', 'VOLUME']]

# we want to 'predict' the future price
# so our label is 'closing' price n day ahead
forecast_col = 'CLOSE'
# fill NaN datas with outliers
df.fillna(value=-99999, inplace=True)
# how many days in the future we want to predict the price
# in percent of length of our data
days_predict_pct = 0.005
forecast_out = int(math.ceil(days_predict_pct * len(df)))
# dates asscociated with above data points
dates_final = dates[-forecast_out:]

# label in each row is the closing price of n days ahead
df['label'] = df[forecast_col].shift(-forecast_out)

# features: all columns but 'label'
X = np.array(df.drop(['label'], axis=1))
# scaling the features so things go a little faster
X = preprocessing.scale(X)
# these are feature set that we won't train classifier
# with them, beacause we don't have 'label' for them
# we will use these for prediction
X_final = X[-forecast_out:]
# final set of features
X = X[:-forecast_out]
# drop any remaining NaN values
df.dropna(inplace=True)
# our labels
y = np.array(df['label'])

# defining our test and training data, 20% of data is set for testing
X_train, X_test, y_train, y_test =  model_selection.train_test_split(X, y, test_size=0.2)

# choose the classifier
clf = LinearRegression(n_jobs=-1)
# different kernels: 'linear', 'poly', 'rbf', 'sigmoid'
# clf = svm.SVR(kernel='linear')
# training classifier
clf.fit(X_train, y_train)
# testing classifier
confidence = clf.score(X_test, y_test)
# our predicted labels for final feature set(X_final)
forecast_set = clf.predict(X_final)

print("\nConfidence Level:> {}".format(confidence))
print("Number of Predicted Days:> {}".format(forecast_out))
print("\nForecasted Prices:> \n{}" .format(forecast_set))

# plotting original and predicted prices
fig, ax = plt.subplots()
ax.set_title("Iran Khodro Daily Closing Price")
ax.set_ylabel("Close Price(IRR)")
ax.set_xlabel("Date")

ax.plot(dates, close_price, 'k', label='Original Prices')
ax.plot(dates_final, forecast_set, 'g--', label='Predicted Prices')

legend = ax.legend()
plt.show()
