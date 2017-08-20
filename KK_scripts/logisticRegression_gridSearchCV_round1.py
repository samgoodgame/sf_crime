# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 12:52:36 2017

@author: kalvi
"""

# Additional Libraries
#%matplotlib inline
import matplotlib.pyplot as plt

# Import relevant libraries:
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# Import Meta-estimators
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Import Calibration tools
from sklearn.calibration import CalibratedClassifierCV

# Set random seed and format print output:
np.random.seed(0)
np.set_printoptions(precision=3)

# Data path to your local copy of Kalvin's "x_data.csv", which was produced by the negated cell above
data_path = "./data/x_data_3.csv"
df = pd.read_csv(data_path, header=0)
x_data = df.drop('category', 1)
y = df.category.as_matrix()

# Impute missing values with mean values:
#x_complete = df.fillna(df.mean())
x_complete = x_data.fillna(x_data.mean())
X_raw = x_complete.as_matrix()

# Scale the data between 0 and 1:
X = MinMaxScaler().fit_transform(X_raw)

# Shuffle data to remove any underlying pattern that may exist.  Must re-run random seed step each time:
np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, y = X[shuffle], y[shuffle]

test_data, test_labels = X[800000:], y[800000:]
dev_data, dev_labels = X[700000:800000], y[700000:800000]
train_data, train_labels = X[:700000], y[:700000]

mini_train_data, mini_train_labels = X[:200000], y[:200000]
mini_dev_data, mini_dev_labels = X[430000:480000], y[430000:480000]

crime_labels = list(set(y))
crime_labels_mini_train = list(set(mini_train_labels))
crime_labels_mini_dev = list(set(mini_dev_labels))
#print(len(crime_labels), len(crime_labels_mini_train), len(crime_labels_mini_dev))

#print(len(train_data),len(train_labels))
#print(len(dev_data),len(dev_labels))
#print(len(mini_train_data),len(mini_train_labels))
#print(len(mini_dev_data),len(mini_dev_labels))
#print(len(test_data),len(test_labels))


#L1
#lr_param_grid_1 = {'C': [0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
lr_param_grid_1 = {'C': [7.5, 10.0, 12.5, 15.0, 20.0, 30.0, 50.0, 100.0]}
LR_l1 = GridSearchCV(LogisticRegression(penalty='l1'), param_grid=lr_param_grid_1, scoring='neg_log_loss')
LR_l1.fit(train_data, train_labels)

print('L1: best C value:', str(LR_l1.best_params_['C']))

LR_l1_prediction_probabilities = LR_l1.predict_proba(dev_data)
LR_l1_predictions = LR_l1.predict(dev_data)
print("L1 Multi-class Log Loss:", log_loss(y_true = dev_labels, y_pred = LR_l1_prediction_probabilities, labels = crime_labels), "\n\n")

#create an LR-L1 classifier with the best params
bestL1 = LogisticRegression(penalty='l1', C=LR_l1.best_params_['C'])
bestL1.fit(train_data, train_labels)
#L1weights = bestL1.coef_

columns = ['hour_of_day','dayofweek',\
          'x','y','bayview','ingleside','northern',\
          'central','mission','southern','tenderloin',\
          'park','richmond','taraval','HOURLYDRYBULBTEMPF',\
          'HOURLYRelativeHumidity','HOURLYWindSpeed',\
          'HOURLYSeaLevelPressure','HOURLYVISIBILITY',\
          'Daylight']

allCoefsL1 = pd.DataFrame(index=columns)
for a in range(len(bestL1.coef_)):
    allCoefsL1[crime_labels[a]] = bestL1.coef_[a]

allCoefsL1

f = plt.figure(figsize=(15,8))
allCoefsL1.plot(kind='bar', figsize=(15,8))
plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
plt.show()


#L2
lr_param_grid_2 = {'C': [0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0], \
                 'solver':['liblinear','newton-cg','lbfgs', 'sag']}
LR_l2 = GridSearchCV(LogisticRegression(penalty='l2'), param_grid=lr_param_grid_2, scoring='neg_log_loss')
LR_l2.fit(train_data, train_labels)

print('L2: best C value:', str(LR_l2.best_params_['C']))
print('L2: best solver:', str(LR_l2.best_params_['solver']))

LR_l2_prediction_probabilities = LR_l2.predict_proba(dev_data)
LR_l2_predictions = LR_l2.predict(dev_data)
print("L2 Multi-class Log Loss:", log_loss(y_true = dev_labels, y_pred = LR_l2_prediction_probabilities, labels = crime_labels), "\n\n")

#create an LR-L2 classifier with the best params
bestL2 = LogisticRegression(penalty='l2', solver=LR_l2.best_params_['solver'], C=LR_l2.best_params_['C'])
bestL2.fit(train_data, train_labels)
#L2weights = bestL2.coef_

columns = ['hour_of_day','dayofweek',\
          'x','y','bayview','ingleside','northern',\
          'central','mission','southern','tenderloin',\
          'park','richmond','taraval','HOURLYDRYBULBTEMPF',\
          'HOURLYRelativeHumidity','HOURLYWindSpeed',\
          'HOURLYSeaLevelPressure','HOURLYVISIBILITY',\
          'Daylight']

allCoefsL2 = pd.DataFrame(index=columns)
for a in range(len(bestL2.coef_)):
    allCoefsL2[crime_labels[a]] = bestL2.coef_[a]

allCoefsL2

f = plt.figure(figsize=(15,8))
allCoefsL2.plot(kind='bar', figsize=(15,8))
plt.legend(loc='center left', bbox_to_anchor=(1.0,0.5))
plt.show()