# Import Keras stuff
from keras.models import Sequential
from keras import layers, Input, models
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf

# Import other stuff
import numpy as np
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from prettytable import PrettyTable

# Custom Libraries
from Model import BinaryClassifier, Predictor
from DataLoading import LoadDataSet, SplitDataSet, GetDimensions, Normalize
from Plot import Plot, RegressionExamplesPlot, InfoPlotForPresentation
from Accuracy import RegressionMeanAccuracy, RegressionPrintExamples, ClassifierReport

# Clear up annoying warnings
import os
clear = lambda: os.system('cls')
clear()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.mode.chained_assignment = None