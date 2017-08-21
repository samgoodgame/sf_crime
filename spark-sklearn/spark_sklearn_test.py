
## Non-optimized:
#
# from sklearn import grid_search, datasets
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.grid_search import GridSearchCV
#
#
# digits = datasets.load_digits()
# X, y = digits.data, digits.target
# param_grid = {"max_depth": [3, None],
#               "max_features": [1, 3, 10],
#               "min_samples_split": [2, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"],
#               "n_estimators": [10, 20, 40, 80]}
# gs = grid_search.GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
# print(gs.fit(X, y))

## Spark-optimized:

print("Spark-optimized grid search:")

from sklearn import grid_search, datasets
from sklearn.ensemble import RandomForestClassifier
# Use spark_sklearn’s grid search instead:
from spark_sklearn import GridSearchCV
digits = datasets.load_digits()
X, y = digits.data, digits.target
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [10, 20, 40, 80]}
gs = grid_search.GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
print(gs.fit(X, y))
