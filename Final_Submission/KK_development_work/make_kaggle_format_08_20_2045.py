# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:49:47 2017

@author: kalvi
"""

#required imports
import pandas as pd
import numpy as np

def make_kaggle_format(sample_submission_path, test_transformed_path, prediction_probabilities):
    """this function requires:
        sample_submission_path=(filepath for 'sampleSubmission.csv')
        test_transformed_path=(filepath for 'test_transformed.csv')
        prediction_probabilities=(output from clf.predict_proba(test_data))"""
    
    #this is for extracting the column names for the required submission format
    sampleDF = pd.read_csv(sample_submission_path)
    allColumns = list(sampleDF.columns)
    featureColumns = allColumns[1:]
    
    #this is for extracting the test data IDs for our baseline submission
    testDF = pd.read_csv(test_transformed_path, header=0)
    testIDs = list(testDF.axes[0])
    
    #first we turn the prediction probabilties into a data frame
    resultDF = pd.DataFrame(prediction_probabilities,columns=featureColumns)
    
    #this adds the IDs as a final column
    resultDF.loc[:,'Id'] = pd.Series(testIDs,index=resultDF.index)
    
    #the next few lines make the 'Id' column the first column
    colnames = resultDF.columns.tolist()
    colnames = colnames[-1:] + colnames[:-1]
    resultDF = resultDF[colnames]
    
    #output to .csv file
    resultDF.to_csv('result.csv',index=False)