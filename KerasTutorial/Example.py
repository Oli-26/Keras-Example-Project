from Imports import *

# Defining the hyper paramaters for the network, these will be useful to have at the top of the file.
epoch_number = 100 # Number of times we train the machine on the data. A low number might cause underfitting, a high number might cause overfitting.
batch_size = 20 # Amount of examples tested on each forward pass, lower number means we can jump out of local minima. A higher number means more controlled moving towards accuracy.
activation = "selu"   # elu, relu, selu
optimizer = "adam" # adam, nadam, adadelta, adamax, RMSprop - This controls how fast the network will learn, as well as other factors.

# Load housing
x_data, y_data = LoadDataSet("regression")

# Normalized input values
normalized_x_data = Normalize(x_data)

# Split it into training, validation, and testing data sets.

# train - Train the model
# val - validatie how well the training is going
# test - Test how well the model performs (at the end)
x_train, y_train, x_val, y_val, x_test, y_test = SplitDataSet(normalized_x_data, y_data)

# Let Create a multi layer perception, in essence, a basic neural network.
## First lets determine the input dimension of our network. This will be the amount of variables each example contains.
input_dimension = normalized_x_data.shape[1]

## Next, lets create our model.
model = Sequential([
            Input(shape = input_dimension),
            layers.Dense(320, activation=activation), # RELU = rectified (exponential) linear unit
            layers.Dense(160, activation=activation),
            layers.Dense(80, activation=activation),
            layers.Dense(40, activation=activation),
            layers.Dense(20, activation=activation),
            layers.Dense(1, activation=activation)
])

## Now we compile the model to use mean squared error as a loss function. 
model.compile(loss='mse',  optimizer=optimizer)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

## Train the model on the training data, using the validation data as an indication of how the machine is performing.
history = model.fit(x_train, y_train, epochs = epoch_number, verbose=True, batch_size = batch_size, validation_data = (x_val, y_val), callbacks=[callback])

# Finally, lets test our model again the test data.
expected_output = y_test["SalePrice"].tolist()

predicted_output = model.predict(x_test).tolist()

RegressionPrintExamples(expected_output, predicted_output, 10)
input("")

RegressionMeanAccuracy(expected_output, predicted_output)
input("")

InfoPlotForPresentation(history, expected_output, predicted_output)
