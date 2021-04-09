import numpy as np
import matplotlib.pyplot as plt

def Plot(history):
    plt.subplot(1, 2, 2)

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.plot(np.arange(len(history.history['loss'])), history.history['loss'], c="red")
    plt.plot(np.arange(len(history.history['val_loss'])), history.history['val_loss'], c="blue")
    plt.show()


def RegressionExamplesPlot(expected_output, predicted_output):
    plt.subplot(1, 2, 1)
    x = expected_output
    y = predicted_output
    plt.scatter(x, y)
    plt.xlim([0, 500000])
    plt.ylim([0, 500000])
    plt.xlabel("actual price")
    plt.ylabel("predicted price")


    m1, b1 = np.polyfit(x, y, 1)

    plt.plot(x, x*m1 + b1, c="red")
    plt.plot([1, 500000], [1,500000], c="green")

def InfoPlotForPresentation(history, expected_output, predicted_output):
    RegressionExamplesPlot(expected_output, predicted_output)
    Plot(history)