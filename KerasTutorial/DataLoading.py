import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical

def LoadDataSet(mode = "regression", splitPoint = 200000):
    df = pd.read_csv('../Data/train.csv', index_col=0)
    dfX = df[["MoSold", "YrSold", "SaleType", "SaleCondition", "GrLivArea",  "YearBuilt", "OverallQual", "OverallCond", "BldgType", "GarageArea"]]

    # Convert sale type & condition to a number
    SaleTypes = dfX.SaleType.unique().tolist()
    SaleConditions = dfX.SaleCondition.unique().tolist()
    dfX.SaleType = dfX.SaleType.map(lambda x: SaleTypes.index(x))
    dfX.SaleCondition = dfX.SaleCondition.map(lambda x: SaleConditions.index(x))

    #other conversions
    BldgType = dfX.BldgType.unique().tolist()
    dfX.BldgType = dfX.BldgType.map(lambda x: BldgType.index(x))

    
    dfY = df[["SalePrice"]]
    # Create binary output to see if house costs more than 500k
    if(mode == "binary"):
        dfY.SalePrice = dfY.SalePrice.map(lambda x: 1 if x > splitPoint else 0)

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