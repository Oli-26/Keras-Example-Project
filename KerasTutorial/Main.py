from Imports import *

def CreateBasicBinaryMLP(epochs, batch_size, network_depth, activation = "relu", optimizer = "adam"):
    x_data, y_data = LoadDataSet("binary")
    x_train, y_train, x_val, y_val, x_test, y_test = SplitDataSet(x_data, y_data, "binary")

    input_dimension, output_dimension = GetDimensions(x_train, y_train)

    neuralModel = BinaryClassifier(input_dimension, output_dimension, network_depth, activation, optimizer)
    neuralModel.TrainModel(x_train, y_train, x_val, y_val, epochs, batch_size)
    neuralModel.TestModel(x_test, y_test)
    Plot(neuralModel.history)

    
def CreateBasicPredictorMLP(epochs, batch_size, network_depth, activation = "relu", optimizer = "adam"):
    x_data, y_data = LoadDataSet("regression")
    x_train, y_train, x_val, y_val, x_test, y_test = SplitDataSet(x_data, y_data, "regression", 152)

    input_dimension, output_dimension = GetDimensions(x_train, y_train)

    neuralModel = Predictor(input_dimension, network_depth, activation, optimizer)
    neuralModel.TrainModel(x_train, y_train, x_val, y_val, epochs, batch_size)
    neuralModel.TestModel(x_test, y_test)
    

epoch_number = 500 # Number of times we train the machine on the data. A low number might cause underfitting, a high number might cause overfitting.
batch_size = 500 # Amount of examples tested on each forward pass, lower number means we can jump out of local minima. A higher number means more controlled moving towards accuracy.
network_width = 100 # Width of layers.
activation = "relu"   # elu, relu, selu
optimizer = "adam" # adam, nadam, adadelta, adamax, RMSprop

#CreateBasicPredictorMLP(epoch_number, batch_size, network_depth, activation, optimizer)
CreateBasicBinaryMLP(epoch_number, batch_size, network_width, activation, optimizer)
