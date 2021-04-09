import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical

def PriceCategories(x):
    if(x > 400000):
        return 4
    if(x > 300000):
        return 3
    if(x > 200000):
        return 2
    if(x > 100000):
        return 1
    return 0


def LoadDataSet(mode = "regression"):
    df = pd.read_csv('../Data/train.csv', index_col=0)
    dfX = df[["MoSold", "YrSold", "SaleType", "SaleCondition", "GrLivArea",  "YearBuilt", "OverallQual", "OverallCond", "BldgType", "GarageArea"]]
    
    # Convert non number values to numbers.
    SaleTypes = dfX.SaleType.unique().tolist()
    SaleConditions = dfX.SaleCondition.unique().tolist()
    dfX.SaleType = dfX.SaleType.map(lambda x: SaleTypes.index(x))
    dfX.SaleCondition = dfX.SaleCondition.map(lambda x: SaleConditions.index(x))
    BldgType = dfX.BldgType.unique().tolist()
    dfX.BldgType = dfX.BldgType.map(lambda x: BldgType.index(x))


    dfY = df[["SalePrice"]]
    if(mode == "binary"):
        dfY.SalePrice = dfY.SalePrice.map(lambda x: PriceCategories(x))
        

    return (dfX, dfY)

def SplitDataSet(x_data, y_data, mode = "regression", random = 42):
    if(mode == "binary"):
        y_data = to_categorical(y_data)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, random_state = random, train_size = 0.6) 
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, random_state = random, train_size = 0.5) 

    return (x_train, y_train, x_val, y_val, x_test, y_test)

def GetDimensions(x_data, y_data):
    input_dimension = x_data.shape[1]
    output_dimension = y_data.shape[1]

    return (input_dimension, output_dimension)

def Normalize(data):
    normalized_data = data.copy()
    for feature_name in data.columns:
        max_value = data[feature_name].max()
        min_value = data[feature_name].min()
        normalized_data[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
    return normalized_data
