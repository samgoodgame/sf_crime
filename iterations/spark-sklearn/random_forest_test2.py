import numpy as np
from time import time
from operator import itemgetter
from sklearn import svm, grid_search
from sklearn.ensemble import RandomForestClassifier
from spark_sklearn import GridSearchCV
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


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


param_grid = {"max_depth": [None],
              "max_features": ['auto', 10, 15],
              "min_samples_split": [1],
              "min_samples_leaf": [10, 50, 80],
              "bootstrap": [True],
              "criterion": ["gini", "entropy"],
              "n_estimators": [80, 100, 150]}
clf = RandomForestClassifier()

gs = GridSearchCV(sc, clf, param_grid) # add "n_jobs?"
start = time()
gs.fit(mini_train_data, mini_train_labels)
print("GridSearchCV took {:.2f} seconds for {:d} candidate settings.".format(time() - start, len(gs.grid_scores_)))
report(gs.grid_scores_)
