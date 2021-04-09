import numpy as np
from sklearn.metrics import classification_report
from prettytable import PrettyTable

def RegressionPrintExamples(expected_output, predicted_output, amount):
    t = PrettyTable(['Actual price', 'Predicted price', 'Accuracy (%)'])
    for i in range(0,amount):
        accuracy = (1-abs(predicted_output[i][0]-expected_output[i])/expected_output[i])*100
        t.add_row([expected_output[i], "%0.0f" % predicted_output[i][0], "%0.2f" % accuracy])
    print(t)

def RegressionMeanAccuracy(expected_output, predicted_output):
    mean_difference_total = 0
    mean_percent_accuracy_total = 0
    mean_price = 0
    for i in range(0, len(expected_output)):
        mean_difference = abs(expected_output[i] - predicted_output[i][0])
        mean_percent_accuracy = 1.00 - mean_difference/expected_output[i]

        mean_difference_total += mean_difference
        mean_percent_accuracy_total += mean_percent_accuracy
        mean_price += expected_output[i]

    mean_price = mean_price/len(expected_output)

    generic_accuracy = 0
    for i in range(0, len(expected_output)):
        mean_difference = abs(expected_output[i] - mean_price)
        generic_accuracy += 1.00 - mean_difference/expected_output[i]

    generic_accuracy = generic_accuracy/len(expected_output)

    mean_difference_total = mean_difference_total/len(expected_output)
    mean_percent_accuracy_total = mean_percent_accuracy_total/len(expected_output)

    t = PrettyTable(['Statistic Desc', 'Value'])
    t.add_row(['Mean house price', "%0.2f" % mean_price])
    t.add_row(['Mean absolute error of predictions',  "%0.2f" % mean_difference_total]) # ACC Is mean dif/mean cost.
    t.add_row(['Mean accuracy of predictions', str( "%0.2f" % (mean_percent_accuracy_total*100)) + "%"])
    t.add_row(['Mean accuracy if mean-price always predicted', str( "%0.2f" % (generic_accuracy*100)) + "%"])
    print(t)


def ClassifierReport(expected_output, predicted_output):
    print(classification_report(expected_output, predicted_output))