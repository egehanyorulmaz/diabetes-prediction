
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def cate_colName(Transformer, category_cols, drop='if_binary'):
    """
    Name creating function for categorical columns after hotencoding 
    
    :param Transformer: hotencoding transformer
    :param category_cols: categorical columns
    :param drop:
    """
    
    cate_cols_new = []
    col_value = Transformer.categories_
    
    for i, j in enumerate(category_cols):
        if (drop == 'if_binary') & (len(col_value[i]) == 2):
            cate_cols_new.append(j)
        else:
            for f in col_value[i]:
                feature_name = j + '_' + f
                cate_cols_new.append(feature_name)
    return(cate_cols_new)

def Binary_Cross_Combination(colNames, X_train, X_test, OneHot=True):
    """
    Cross combination of binary categorical features
    
    :param colNames: column names 
    :X_train: train dataset
    :X_test: test dataset
    :param OneHot: hotencoding
    
    :return：new feature generated and new names
    """
    
    # create empty storer
    colNames_new_l = []
    features_new_l_train = []
    features_new_l_test = []
    
    # extracting features
    features = X_train[colNames]
    
    # creating new features and new names
    for col_index, col_name in enumerate(colNames):
        for col_sub_index in range(col_index+1, len(colNames)):
            
            newNames = col_name + '&' + colNames[col_sub_index]
            colNames_new_l.append(newNames)
            
            newDF_train = pd.Series(X_train[col_name].astype('str')  
                              + '&'
                              + X_train[colNames[col_sub_index]].astype('str'), 
                              name=col_name)
            newDF_test = pd.Series(X_test[col_name].astype('str')  
                              + '&'
                              + X_test[colNames[col_sub_index]].astype('str'), 
                              name=col_name)
            features_new_l_train.append(newDF_train)
            features_new_l_test.append(newDF_test)
    
    # Connecting new feature matrix
    features_new_train = pd.concat(features_new_l_train, axis=1)
    features_new_train.columns = colNames_new_l
        
    features_new_test = pd.concat(features_new_l_test, axis=1)
    features_new_test.columns = colNames_new_l
    
    colNames_new = colNames_new_l
    
    # Hotencoding for new features
    if OneHot == True:
        enc1 = OneHotEncoder()
        enc1.fit_transform(features_new_train)
        colNames_new = cate_colName(enc1, colNames_new_l, drop=None)
        CrossComb_train = pd.DataFrame(enc1.fit_transform(features_new_train).toarray(), columns=colNames_new)
        
        enc2 = OneHotEncoder()
        enc2.fit_transform(features_new_test)
        colNames_new = cate_colName(enc2, colNames_new_l, drop=None)
        CrossComb_test = pd.DataFrame(enc2.fit_transform(features_new_test).toarray(), columns=colNames_new)
    
    return CrossComb_train, CrossComb_test, colNames_new


def Combination(colNames, features, OneHot=True):
    """
    a cross combination method

    :param colNames: feature names array
    :param features: original dataset
    :param OneHot: hotencoding boolean

    :return：cross-combined features and name list
    """

    # create empty arrays for storage
    colNames_new_l = []
    features_new_l = []

    # feature extraction
    features = features[colNames]

    # create new features and names
    for col_index, col_name in enumerate(colNames):
        for col_sub_index in range(col_index+1, len(colNames)):
            
            newNames = col_name + '&' + colNames[col_sub_index]
            colNames_new_l.append(newNames)
            
            newDF = pd.Series(features[col_name].astype('str')  
                                + '&'
                                + features[colNames[col_sub_index]].astype('str'), 
                                name=col_name)
            features_new_l.append(newDF)

    # connect new feature matrix
    features_new = pd.concat(features_new_l, axis=1)
    features_new.columns = colNames_new_l
    colNames_new = colNames_new_l

    # hot encoding
    if OneHot == True:
        enc = preprocessing.OneHotEncoder()
        enc.fit_transform(features_new)
        colNames_new = cate_colName(enc, colNames_new_l, drop=None)
        features_new = pd.DataFrame(enc.fit_transform(features_new).toarray(), columns=colNames_new)
        
    return features_new, colNames_new