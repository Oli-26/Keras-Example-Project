from keras.models import Sequential
from keras import layers, Input, models
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import classification_report

from Accuracy import RegressionMeanAccuracy, RegressionPrintExamples, ClassifierReport
from Plot import Plot, RegressionExamplesPlot
class BinaryClassifier:
    def __init__(self, input_dimension, output_dimension, layer_amount, activation = "relu", optimizer = "adam"):
        self.CreateModel(input_dimension, output_dimension, layer_amount, activation, optimizer)

    def CreateModel(self, input_dimension, output_dimension, layer_amount, activation, optimizer):
        self.model = Sequential()
        self.model.add(layers.Dense(layer_amount, activation=activation, input_dim = input_dimension))
        self.model.add(layers.Dense(output_dimension, activation='softmax'))
        self.model.compile(loss='binary_crossentropy',  optimizer=optimizer)
    
    def TrainModel(self, x_train, y_train, x_val, y_val, epochs = 15, batch_size = 15):
        self.history = self.model.fit(x_train, y_train, epochs = epochs, verbose=True, batch_size = batch_size, validation_data = (x_val, y_val))
        print("Done.")
    
    def TestModel(self, x_test, y_test):
        expected_output = np.array([np.argmax(x) for x in y_test])
        predicted_output = np.array([np.argmax(x) for x in self.model.predict(x_test)])
        ClassifierReport(expected_output, predicted_output)


class Predictor:
    def __init__(self, input_dimension, layer_amount, activation = "relu", optimizer = "adam"):
        self.CreateModel(input_dimension, layer_amount, activation, optimizer)

    def CreateModel(self, input_dimension, layer_amount, activation, optimizer):
        self.model = Sequential()
        self.model.add(layers.Dense(layer_amount, activation=activation, input_dim = input_dimension))
        self.model.add(layers.Dense(1, activation=activation))
        self.model.compile(loss='mse',  optimizer=optimizer)
    
    def TrainModel(self, x_train, y_train, x_val, y_val, epochs = 15, batch_size = 15):
        self.history = self.model.fit(x_train, y_train, epochs = epochs, verbose=True, batch_size = batch_size, validation_data = (x_val, y_val))
        print("Done.")
    
    def TestModel(self, x_test, y_test):        
        expected_output = y_test["SalePrice"].tolist()
        predicted_output = self.model.predict(x_test).tolist()
        RegressionPrintExamples(expected_output, predicted_output, 10)
        RegressionMeanAccuracy(expected_output, predicted_output)
        RegressionExamplesPlot(expected_output, predicted_output)
