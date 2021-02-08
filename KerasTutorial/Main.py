# Import Keras stuff
from keras.models import Sequential
from keras import layers, Input, models
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# Import other stuff
import numpy as np
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Clear up annoying warnings
import os
clear = lambda: os.system('cls')
clear()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None


from Model import BinaryClassifier, Predictor
from DataLoading import LoadDataSet, SplitDataSet, GetDimensions

def CreateBasicBinaryMLP(epochs, batch_size, network_depth):
    x_data, y_data = LoadDataSet("binary")
    x_train, y_train, x_val, y_val, x_test, y_test = SplitDataSet(x_data, y_data, "binary")

    input_dimension, output_dimension = GetDimensions(x_train, y_train)

    neuralModel = BinaryClassifier(input_dimension, output_dimension, network_depth)
    neuralModel.TrainModel(x_train, y_train, x_val, y_val, epochs, batch_size)
    neuralModel.TestModel(x_test, y_test)
    
def CreateBasicPredictorMLP(epochs, batch_size, network_depth):
    x_data, y_data = LoadDataSet("regression")
    x_train, y_train, x_val, y_val, x_test, y_test = SplitDataSet(x_data, y_data, "regression")

    input_dimension, output_dimension = GetDimensions(x_train, y_train)

    neuralModel = Predictor(input_dimension, network_depth)
    neuralModel.TrainModel(x_train, y_train, x_val, y_val, epochs, batch_size)
    neuralModel.TestModel(x_test, y_test)



epoch_number = 1000
batch_size = 50
network_depth = 100


#CreateBasicPredictorMLP(epoch_number, batch_size, network_depth)
CreateBasicBinaryMLP(epoch_number, batch_size, network_depth)
