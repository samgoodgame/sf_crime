# Predicting Crime in San Francisco

### Sarah Cha, Sam Goodgame, Kalvin Kao, and Bryan Moore

[Bay Bridge by Sam Goodgame](bay_bridge_goodgame.jpg)


Our goal with this project is to predict the type of a crime based on its component details. The component details are primarily related to time and location. The data is from a [Kaggle competition](https://www.kaggle.com/c/sf-crime/discussion).

This is an interesting problem space because time and location are both high-dimensional variables. Such variables don't tend to work well with machine learning models, because they lead the models to overfit and generalize poorly.

Accordingly, our goal in this project is to generate an accurate, parsimonious model by working our way through the model selection triple:
- Model selection
- Feature Engineering
- Hyperparameter tuning (and calibration)

While the model selection triple may seem like a linear checklist, we did not approach it that way. In other words, we conducted model selection, feature engineering, and hyperparameter tuning in parallel, with different members of the team focusing on different aspects of the problem at the same time.

After transforming the data into a usable format, we set about engineering useful features. We focused primarily on enriching our data with features related to weather and schools.

Then, we prototyped the major model types using their default specifications. Then, we dug slightly deeper into each model type by tuning the hyperparameters and calibrating the models.

View the project presentation [here](https://docs.google.com/a/berkeley.edu/presentation/d/1JcpZkVXQVwGmCtJR1yn8Rz-LG-JWqjNKl1N2u7wcWV8/edit?usp=sharing).

## Results

We were able to achieve a multi-class logarithmic loss of 2.37 on development data. The winning model was a random forest classifier with isotonic calibration.

## Using This Repository

The code for our winning model is in the notebook `predicting_crimes.ipynb`.
We analyzed the errors from that model in the notebook `error_analysis.ipynb`.
We trained lots of models (and thousands of model specifications). The work associated with those models is in the notebook `supporting_notebook.ipynb`.

## Using Spark for GridSearchCV

We parallelized our grid search using Spark. To do this, we created an EC2 instance with Spark, Java, Scala, Python, Pip, and relevant scientific computing packages. We then ran our gridsearch using a SparkContext, which allowed us to decrease our hyperparameter tuning runtime to manageable blocks.

Here is an example block of code that we used to run an iteration of hyperparameter tuning (in this case, for a k-nearest neighbors classifier):

```python
import numpy as np
from time import time
from operator import itemgetter
from sklearn import svm, grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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


param_grid = {"n_neighbors": ([i for i in range(1,21,3)] + [j for j in range(25,51,10)] + [k for k in range(55,13300,3000)]),
              "weights": ['uniform', 'distance'],
              "p": [1,2]}
clf = KNeighborsClassifier()

gs = GridSearchCV(sc, clf, param_grid)
start = time()
gs.fit(mini_train_data, mini_train_labels)
print("GridSearchCV took {:.2f} seconds for {:d} candidate settings.".format(time() - start, len(gs.grid_scores_)))
report(gs.grid_scores_)
```
