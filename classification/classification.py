import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

'''
based on a set of attributes of a website we want to know
if it is a likely a phishing website or not.

data taken from "https://archive.ics.uci.edu/ml/datasets/Phishing+Websites"
for a description of dataset refer to above link
'''

# load arff file into numpy record array
# meta is the header information(attributes,...)
data, meta = loadarff('phishing_sites.arff')
# create a pandas dataframe from record array
# it is easier to performe any action on data
# in pandas dataframe, though here isn't much!
df = pd.DataFrame.from_records(data, coerce_float=True)
# convert boolean type of columns to int
for col in df.columns.values:
	df[col] = df[col].astype(int)

# data doesn't contain missing data, just to be sure
# drop any NaN values
df.dropna(inplace=True)

# feature set: anything but 'Result'
X = np.array(df.drop(['Result'], axis=1))
# label set: 'Result' column
y = np.array(df['Result'])

# split training data from testing one
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# defining classifier
clf = neighbors.KNeighborsClassifier()
# train the classifier
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print("\nAccuracy: {}\n" .format(accuracy))

# a prediction example:
sample_site_measures = np.array([-1,0,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,0,-1,1,1,0,1,1,1,1,-1,1,-1,-1,-1,1,1])
sample_site_measures = sample_site_measures.reshape(1, -1)

result = clf.predict(sample_site_measures)
if result == 1:
	result = 'Safe'
elif result == -1:
	result = 'Unsafe'

print("\nPrediction for Site Measures {}\n is {}" .format(sample_site_measures, result))