# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:47:56 2016

@author: Harminder
"""

#import numpy as np
import pandas as pd

train_set = pd.read_csv("C:\\Users\\Harminder\\Desktop\\project\\home depot\\dataset\\train.csv", encoding="ISO-8859-1")

attribute = pd.read_csv("C:\\Users\\Harminder\\Desktop\\project\\home depot\\dataset\\attributes.csv", encoding="ISO-8859-1")

test = pd.read_csv("C:\\Users\\Harminder\\Desktop\\project\\home depot\\dataset\\test.csv", encoding="ISO-8859-1")

pro_desc = pd.read_csv("C:\\Users\\Harminder\\Desktop\\project\\home depot\\dataset\\description_modified.csv", encoding="ISO-8859-1")



#Summary of train data

train_set.info()

train_set.relevance.hist(color='c', alpha=0.5) #histogram of relevance score

train_set.relevance.value_counts()  #unique frequency score and Frequency count of them 

train_set.product_uid.nunique()  # Unique Product id in train

train_set.shape #Dimension of train set

#summary of test data

test.info()

test.product_uid.nunique() #Unique Product id in test data

test.shape #dimension of test set

#summary of attribute file

attribute.name.value_counts() #Extracting attributes from name column

attribute.shape #dimension of atttribute set


#finding missing value in attribute file
attribute_lenght_of_rows=attribute.shape[0]

without_missing_values=attribute.count()[1]

total_missing=attribute_lenght_of_rows-without_missing_values

percentage_missin_atttribute=15500/2044803 #percentage of missing values


#Summary of product description

pro_desc.info()



