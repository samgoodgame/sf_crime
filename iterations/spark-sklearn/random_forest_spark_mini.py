# Parallelized gridsearch using a package that integrates Spark with scikit-learn.
# I couldn't run this in AWS EC2 because I couldn't manage to download
# the right versions of Spark, PySpark, Python, Pip, and the dependencies all together.
# It's definitely possible, just beyond my current linux abilities.

from sklearn import grid_search
from sklearn.ensemble import RandomForestClassifier
from spark_sklearn import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

##################### Data Wrangling #######################################

data_path = "./x_data_3.csv"
df = pd.read_csv(data_path, header=0)
x_data = df.drop('category', 1)
y = df.category.as_matrix()

x_complete = x_data.fillna(x_data.mean())
X_raw = x_complete.as_matrix()
X = MinMaxScaler().fit_transform(X_raw)
np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, y = X[shuffle], y[shuffle]

# Due to difficulties with log loss and set(y_pred) needing to match set(labels), we will remove the extremely rare
# crimes from the data for quality issues.
X_minus_trea = X[np.where(y != 'TREA')]
y_minus_trea = y[np.where(y != 'TREA')]
X_final = X_minus_trea[np.where(y_minus_trea != 'PORNOGRAPHY/OBSCENE MAT')]
y_final = y_minus_trea[np.where(y_minus_trea != 'PORNOGRAPHY/OBSCENE MAT')]

# Separate training, dev, and test data:
test_data, test_labels = X_final[800000:], y_final[800000:]
dev_data, dev_labels = X_final[700000:800000], y_final[700000:800000]
train_data, train_labels = X_final[100000:700000], y_final[100000:700000]
calibrate_data, calibrate_labels = X_final[:100000], y_final[:100000]

# Create mini versions of the above sets
mini_train_data, mini_train_labels = X_final[:20000], y_final[:20000]
mini_calibrate_data, mini_calibrate_labels = X_final[19000:28000], y_final[19000:28000]
mini_dev_data, mini_dev_labels = X_final[49000:60000], y_final[49000:60000]


##################### Grid Search #######################################

param_grid = {"max_depth": [3, None],
              "max_features": [1, 3],
              "min_samples_split": [2, 3],
              "min_samples_leaf": [1, 3],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [3, 4]}
gs = grid_search.GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
gs.fit(mini_train_data, mini_train_labels)
