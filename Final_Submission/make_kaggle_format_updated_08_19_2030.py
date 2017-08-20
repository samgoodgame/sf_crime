# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:49:47 2017

@author: kalvi
"""

#required imports
import pandas as pd
import numpy as np

def make_kaggle_format(sample_submission_path, test_IDs, prediction_probabilities):
    #this is for extracting the column names for the required submission format
    sampleDF = pd.read_csv(sample_submission_path)
    allColumns = list(sampleDF.columns)
    featureColumns = allColumns[1:]
    
    #first we turn the prediction probabilties into a data frame
    resultDF = pd.DataFrame(prediction_probabilities,columns=featureColumns)
    
    #this adds the IDs as a final column
    resultDF.loc[:,'Id'] = pd.Series(test_IDs,index=resultDF.index)
    
    #the next few lines make the 'Id' column the first column
    colnames = resultDF.columns.tolist()
    colnames = colnames[-1:] + colnames[:-1]
    resultDF = resultDF[colnames]
    
    #output to .csv file
    resultDF.to_csv('result.csv',index=False)