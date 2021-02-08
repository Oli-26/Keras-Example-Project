from keras.models import Sequential
from keras import layers, Input, models
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import classification_report

class BinaryClassifier:
    def __init__(self, input_dimension, output_dimension, layer_amount):
        self.CreateModel(input_dimension, output_dimension, layer_amount)

    def CreateModel(self, input_dimension, output_dimension, layer_amount):
        self.model = Sequential()
        self.model.add(layers.Dense(layer_amount, activation='relu', input_dim = input_dimension))
        self.model.add(layers.Dense(output_dimension, activation='softmax'))
        self.model.compile(loss='binary_crossentropy',  optimizer='adam')
    
    def TrainModel(self, x_train, y_train, x_val, y_val, epochs = 15, batch_size = 15):
        self.model.fit(x_train, y_train, epochs = epochs, verbose=True, batch_size = batch_size, validation_data = (x_val, y_val))
        print("Done.")
    
    def TestModel(self, x_test, y_test):
        expected_output = np.array([np.argmax(x) for x in y_test])
        predicted_output = np.array([np.argmax(x) for x in self.model.predict(x_test)])
        
        labels = ['below 200k', 'above 200k']
        print(classification_report(expected_output, predicted_output, target_names = labels))


class Predictor:
    def __init__(self, input_dimension, layer_amount):
        self.CreateModel(input_dimension, layer_amount)

    def CreateModel(self, input_dimension, layer_amount):
        self.model = Sequential()
        self.model.add(layers.Dense(layer_amount, activation='relu', input_dim = input_dimension))
        self.model.add(layers.Dense(1, activation='relu'))
        self.model.compile(loss='mse',  optimizer='adam')
    
    def TrainModel(self, x_train, y_train, x_val, y_val, epochs = 15, batch_size = 15):
        self.model.fit(x_train, y_train, epochs = epochs, verbose=True, batch_size = batch_size, validation_data = (x_val, y_val))
        print("Done.")
    
    def TestModel(self, x_test, y_test):        
        expected_output = y_test["SalePrice"].tolist()
        predicted_output = self.model.predict(x_test).tolist()

        print("Examples (Actual price | predicted price):")
        for i in range(0,25):
            print("         " + str(expected_output[i]) + " | " + str(predicted_output[i]))
        
        print("\n")

        mean_difference_total = 0
        mean_percent_accuracy_total = 0
        for i in range(0, len(expected_output)):
            mean_difference = abs(expected_output[i] - predicted_output[i][0])
            mean_percent_accuracy = 1.00 - mean_difference/expected_output[i]

            mean_difference_total += mean_difference
            mean_percent_accuracy_total += mean_percent_accuracy

        mean_difference_total = mean_difference_total/len(expected_output)
        mean_percent_accuracy_total = mean_percent_accuracy_total/len(expected_output)
        print("Mean difference: " + str(mean_difference_total))
        print("Mean Accuracy: " + str(mean_percent_accuracy_total*100) + "%")