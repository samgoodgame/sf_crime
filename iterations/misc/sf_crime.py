
# coding: utf-8

# # SF Crime
# ## W207 Final Project
# ### Basic Modeling
# 
# 

# ### Environment and Data

# In[1]:


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
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Set random seed and format print output:
np.random.seed(0)
np.set_printoptions(precision=3)


# #### DDL to construct table for SQL transformations:
# 
# ```sql
# CREATE TABLE kaggle_sf_crime (
# dates TIMESTAMP,                                
# category VARCHAR,
# descript VARCHAR,
# dayofweek VARCHAR,
# pd_district VARCHAR,
# resolution VARCHAR,
# addr VARCHAR,
# X FLOAT,
# Y FLOAT);
# ```
# #### Getting training data into a locally hosted PostgreSQL database:
# ```sql
# \copy kaggle_sf_crime FROM '/Users/Goodgame/Desktop/MIDS/207/final/sf_crime_train.csv' DELIMITER ',' CSV HEADER;
# ```
# 
# #### SQL Query used for transformations:
# 
# ```sql
# SELECT
#   category,
#   date_part('hour', dates) AS hour_of_day,
#   CASE
#     WHEN dayofweek = 'Monday' then 1
#     WHEN dayofweek = 'Tuesday' THEN 2
#     WHEN dayofweek = 'Wednesday' THEN 3
#     WHEN dayofweek = 'Thursday' THEN 4
#     WHEN dayofweek = 'Friday' THEN 5
#     WHEN dayofweek = 'Saturday' THEN 6
#     WHEN dayofweek = 'Sunday' THEN 7
#   END AS dayofweek_numeric,
#   X,
#   Y,
#   CASE
#     WHEN pd_district = 'BAYVIEW' THEN 1
#     ELSE 0
#   END AS bayview_binary,
#     CASE
#     WHEN pd_district = 'INGLESIDE' THEN 1
#     ELSE 0
#   END AS ingleside_binary,
#     CASE
#     WHEN pd_district = 'NORTHERN' THEN 1
#     ELSE 0
#   END AS northern_binary,
#     CASE
#     WHEN pd_district = 'CENTRAL' THEN 1
#     ELSE 0
#   END AS central_binary,
#     CASE
#     WHEN pd_district = 'BAYVIEW' THEN 1
#     ELSE 0
#   END AS pd_bayview_binary,
#     CASE
#     WHEN pd_district = 'MISSION' THEN 1
#     ELSE 0
#   END AS mission_binary,
#     CASE
#     WHEN pd_district = 'SOUTHERN' THEN 1
#     ELSE 0
#   END AS southern_binary,
#     CASE
#     WHEN pd_district = 'TENDERLOIN' THEN 1
#     ELSE 0
#   END AS tenderloin_binary,
#     CASE
#     WHEN pd_district = 'PARK' THEN 1
#     ELSE 0
#   END AS park_binary,
#     CASE
#     WHEN pd_district = 'RICHMOND' THEN 1
#     ELSE 0
#   END AS richmond_binary,
#     CASE
#     WHEN pd_district = 'TARAVAL' THEN 1
#     ELSE 0
#   END AS taraval_binary
# FROM kaggle_sf_crime;
# ```

# #### Load the data into training, development, and test:

# In[2]:


data_path = "./data/train_transformed.csv"

df = pd.read_csv(data_path, header=0)
x_data = df.drop('category', 1)
y = df.category.as_matrix()

# Impute missing values with mean values:
x_complete = x_data.fillna(x_data.mean())
X_raw = x_complete.as_matrix()

# Scale the data between 0 and 1:
X = MinMaxScaler().fit_transform(X_raw)

# Shuffle data to remove any underlying pattern that may exist:
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, y = X[shuffle], y[shuffle]

# Separate training, dev, and test data:
test_data, test_labels = X[800000:], y[800000:]
dev_data, dev_labels = X[700000:800000], y[700000:800000]
train_data, train_labels = X[:700000], y[:700000]

mini_train_data, mini_train_labels = X[:75000], y[:75000]
mini_dev_data, mini_dev_labels = X[75000:100000], y[75000:100000]


# In[3]:


#the submission format requires that we list the ID of each example?
#this is to remember the order of the IDs after shuffling
#(not used for anything right now)
allIDs = np.array(list(df.axes[0]))
allIDs = allIDs[shuffle]

testIDs = allIDs[800000:]
devIDs = allIDs[700000:800000]
trainIDs = allIDs[:700000]

#this is for extracting the column names for the required submission format
sampleSubmission_path = "./data/sampleSubmission.csv"
sampleDF = pd.read_csv(sampleSubmission_path)
allColumns = list(sampleDF.columns)
featureColumns = allColumns[1:]

#this is for extracting the test data for our baseline submission
real_test_path = "./data/test_transformed.csv"
testDF = pd.read_csv(real_test_path, header=0)
real_test_data = testDF

test_complete = real_test_data.fillna(real_test_data.mean())
Test_raw = test_complete.as_matrix()

TestData = MinMaxScaler().fit_transform(Test_raw)

#here we remember the ID of each test data point
#(in case we ever decide to shuffle the test data for some reason)
testIDs = list(testDF.axes[0])


# In[4]:


#copied the baseline classifier from below,
#but made it return prediction probabilities for the actual test data
def MNB():
    mnb = MultinomialNB(alpha = 0.0000001)
    mnb.fit(train_data, train_labels)
    #print("\n\nMultinomialNB accuracy on dev data:", mnb.score(dev_data, dev_labels))
    return mnb.predict_proba(real_test_data)
MNB()

baselinePredictionProbabilities = MNB()

#here is my rough attempt at putting the results (prediction probabilities)
#in a .csv in the required format
#first we turn the prediction probabilties into a data frame
resultDF = pd.DataFrame(baselinePredictionProbabilities,columns=featureColumns)
#this adds the IDs as a final column
resultDF.loc[:,'Id'] = pd.Series(testIDs,index=resultDF.index)
#the next few lines make the 'Id' column the first column
colnames = resultDF.columns.tolist()
colnames = colnames[-1:] + colnames[:-1]
resultDF = resultDF[colnames]
#output to .csv file
resultDF.to_csv('result.csv',index=False)


# In[6]:


## Data sanity checks
print(train_data[:1])
print(train_labels[:1])


# In[7]:


# Modeling sanity check with MNB--fast model


def MNB():
    mnb = MultinomialNB(alpha = 0.0000001)
    mnb.fit(train_data, train_labels)
    print("\n\nMultinomialNB accuracy on dev data:", mnb.score(dev_data, dev_labels))
    
MNB()


# ### Model Prototyping
# Rapidly assessing the viability of different model forms:

# In[ ]:


def model_prototype(train_data, train_labels, eval_data, eval_labels):
    knn = KNeighborsClassifier(n_neighbors=5).fit(train_data, train_labels)
    bnb = BernoulliNB(alpha=1, binarize = 0.5).fit(train_data, train_labels)
    mnb = MultinomialNB().fit(train_data, train_labels)
    log_reg = LogisticRegression().fit(train_data, train_labels)
    support_vm = svm.SVC().fit(train_data, train_labels)
    neural_net = MLPClassifier().fit(train_data, train_labels)
    random_forest = RandomForestClassifier().fit(train_data, train_labels)
    
    models = [knn, bnb, mnb, log_reg, support_vm, neural_net, random_forest]
    for model in models:
        eval_preds = model.predict(eval_data)
        print(model, "Accuracy:", np.mean(eval_preds==eval_labels), "\n\n")

model_prototype(mini_train_data, mini_train_labels, mini_dev_data, mini_dev_labels)


# ### K-Nearest Neighbors

# In[ ]:


# def k_neighbors(k_values):
    
#     accuracies = []
#     for k in k_values:
#         clfk = KNeighborsClassifier(n_neighbors=k).fit(train_data, train_labels)
#         dev_preds = clfk.predict(dev_data)
#         accuracies.append(np.mean(dev_preds == dev_labels))
#         print("k=",k, "accuracy:", np.mean(dev_preds == dev_labels))
#         if k == 7: 
#             print("\n\n Classification report for k = 7", ":\n", 
#                   classification_report(dev_labels, dev_preds),)
            
# k_values = [i for i in range(7,9)]

# k_neighbors(k_values)


# ### Multinomial, Bernoulli, and Gaussian Naive Bayes

# In[5]:


def GNB():
    gnb = GaussianNB()
    gnb.fit(train_data, train_labels)
    print("GaussianNB accuracy on dev data:", 
          gnb.score(dev_data, dev_labels))
    
    # Gaussian Naive Bayes requires the data to have a relative normal distribution. Sometimes
    # adding noise can improve performance by making the data more normal:
    train_data_noise = np.random.rand(train_data.shape[0],train_data.shape[1])
    modified_train_data = np.multiply(train_data,train_data_noise)    
    gnb_noise = GaussianNB()
    gnb.fit(modified_train_data, train_labels)
    print("GaussianNB accuracy with added noise:", 
          gnb.score(dev_data, dev_labels))    
    
# Going slightly deeper with hyperparameter tuning and model calibration:
def BNB(alphas):
    
    bnb_one = BernoulliNB(binarize = 0.5)
    bnb_one.fit(train_data, train_labels)
    print("\n\nBernoulli Naive Bayes accuracy when alpha = 1 (the default value):",
          bnb_one.score(dev_data, dev_labels))
    
    bnb_zero = BernoulliNB(binarize = 0.5, alpha=0)
    bnb_zero.fit(train_data, train_labels)
    print("BNB accuracy when alpha = 0:", bnb_zero.score(dev_data, dev_labels))
    
    bnb = BernoulliNB(binarize=0.5)
    clf = GridSearchCV(bnb, param_grid = alphas)
    clf.fit(train_data, train_labels)
    print("Best parameter for BNB on the dev data:", clf.best_params_)
    
    clf_tuned = BernoulliNB(binarize = 0.5, alpha=0.00000000000000000000001)
    clf_tuned.fit(train_data, train_labels)
    print("Accuracy using the tuned Laplace smoothing parameter:", 
          clf_tuned.score(dev_data, dev_labels), "\n\n")
    

def investigate_model_calibration(buckets, correct, total):
    clf_tuned = BernoulliNB(binarize = 0.5, alpha=0.00000000000000000000001)
    clf_tuned.fit(train_data, train_labels)
    
    # Establish data sets
    pred_probs = clf_tuned.predict_proba(dev_data)
    max_pred_probs = np.array(pred_probs.max(axis=1))
    preds = clf_tuned.predict(dev_data)
        
    # For each bucket, look at the predictions that the model yields. 
    # Keep track of total & correct predictions within each bucket.
    bucket_bottom = 0
    bucket_top = 0
    for bucket_index, bucket in enumerate(buckets):
        bucket_top = bucket
        for pred_index, pred in enumerate(preds):
            if (max_pred_probs[pred_index] <= bucket_top) and (max_pred_probs[pred_index] > bucket_bottom):
                total[bucket_index] += 1
                if preds[pred_index] == dev_labels[pred_index]:
                    correct[bucket_index] += 1
        bucket_bottom = bucket_top

def MNB():
    mnb = MultinomialNB(alpha = 0.0000001)
    mnb.fit(train_data, train_labels)
    print("\n\nMultinomialNB accuracy on dev data:", mnb.score(dev_data, dev_labels))

alphas = {'alpha': [0.00000000000000000000001, 0.0000001, 0.0001, 0.001, 
                    0.01, 0.1, 0.0, 0.5, 1.0, 2.0, 10.0]}
buckets = [0.5, 0.9, 0.99, 0.999, .9999, 0.99999, 1.0]
correct = [0 for i in buckets]
total = [0 for i in buckets]

MNB()
GNB()
BNB(alphas)
investigate_model_calibration(buckets, correct, total)

for i in range(len(buckets)):
   accuracy = 0.0
   if (total[i] > 0): accuracy = correct[i] / total[i]
   print('p(pred) <= %.13f    total = %3d    accuracy = %.3f' %(buckets[i], total[i], accuracy))


# The Bernoulli Naive Bayes and Multinomial Naive Bayes models can predict whether a loan will be good or bad with XXX% accuracy.
# 
# ###### Hyperparameter tuning:
# The optimal Laplace smoothing parameter $\alpha$ for the Bernoulli NB model:
# 
# ###### Model calibration:
# Notes
# 

# ### Final evaluation on test data

# In[14]:


def model_test(train_data, train_labels, eval_data, eval_labels):
    '''Similar to the initial model prototyping, but using the 
    tuned parameters on data that none of the models have yet
    encountered.'''
    knn = KNeighborsClassifier(n_neighbors=7).fit(train_data, train_labels)
    bnb = BernoulliNB(alpha=0.0000000000000000000001, binarize = 0.5).fit(train_data, train_labels)
    
    models = [knn, bnb]
    for model in models:
        eval_preds = model.predict(eval_data)
        print(model, "Accuracy:", np.mean(eval_preds==eval_labels), "\n\n")

model_test(train_data, train_labels, test_data, test_labels)

