import numpy as np
from sklearn.metrics import classification_report



def RegressionPrintExamples(expected_output, predicted_output, amount):
    print("Examples (Actual price | predicted price):")
    for i in range(0,amount):
        print("         " + str(expected_output[i]) + " | " + str(predicted_output[i]))
    
    print("\n")

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
    print("Mean difference in model predictions: " + str(mean_difference_total))
    print("Mean Accuracy in model predictions: " + str(mean_percent_accuracy_total*100) + "%")
    print("\n")
    print("Mean house price: " + str(mean_price))
    print("Accuracy if we always guess the mean: " + str(generic_accuracy*100) + "%")


def ClassifierReport(expected_output, predicted_output):
    labels = ['below mark', 'above mark']
    print(classification_report(expected_output, predicted_output, target_names = labels))