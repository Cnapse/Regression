#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as sk


# In[46]:


def load_dataset_from_csv(train_file,test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    return(train,test)


# In[47]:


def data_label(data):
    train = data.iloc[:,1:80]
    label = data.iloc[:,-1]
    return train,label


# In[48]:


def ordinal_encoding(train,test):
    oe = sk.OrdinalEncoder()
    oe.fit(train)
    train_enc = oe.transform(train)
    oe.fit(test)
    test_enc = oe.transform(test)
    return pd.DataFrame(train_enc,columns = train.columns), pd.DataFrame(test_enc,columns = test.columns)


# In[49]:


def fillna(data):
    #add a code to remove the column if the mode is NAn
    #replace NAn with mode in other columns
    data = data.fillna(data.median())
    return data


# In[50]:


def data_cleaner(train_file, test_file):
    train,test = load_dataset_from_csv(train_file,test_file)
    X_train,y_train = data_label(train)
    X_train_enc, X_test_enc = ordinal_encoding(X_train,test)
    X_train_enc, X_test_enc = fillna(X_train_enc), fillna(X_test_enc)
    
    return(X_train_enc,y_train,X_test_enc)
    

