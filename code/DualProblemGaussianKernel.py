"""
    DUAL REGRESSION PROBLEM USING GAUSSIAN KERNEL & GRAM MATRIX
    This program is used to predict the TARGET values, without calculating the THETA
    IT calculates the similarity between features and predicts the target accordingly
    EUCLIDEAN Distance measure is used to calculate the Similarity
"""

import scipy
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen


# KERNEL FUNCTION
def kernel(Xi, X, sigma):
    return scipy.exp(-((np.power(np.linalg.norm(Xi-X),2))/(2*(sigma*sigma))))

# DUAL REGRESSION - GAUSSIAN KERNEL
def DualProblemSolution(trainingSetArray, testSetArray):

    # Take the Traning set Feature and Target value
    X = trainingSetArray[0]
    Y = trainingSetArray[1]

    # Sigma value to set
    sigma = 1

    #TIME PERFORMANCE

    StartTime = time.time()

    # Euclidean Distance
    euclidean_distance = pairwise_distances(X,X, metric='euclidean')
    # Gram Matrix
    GramMatrix = scipy.exp(-euclidean_distance/(2*sigma**2))
    #print(GramMatrix)

    # CALCULATE ALPHA
    ALPHA = np.linalg.solve(GramMatrix, Y)
    #print(ALPHA)

    # Predict the Values for TEST FEATURES
    yHat = []
    for x in testSetArray[0]:
        predict_yHat = 0
        for i in range(X.shape[0]):
           predict_yHat = predict_yHat + (ALPHA[i].item(0) * kernel(X[i],x,sigma))
        yHat.append(predict_yHat)

    print("THE TARGET VALUES ARE: ",yHat)
    print("MSE TEST DATA: ", mean_squared_error(testSetArray[1], yHat))
    print("Time Taken is: ",time.time()-StartTime," seconds")


# Performing 10- Fold cross validation
def crossValidation(var):
    for i in range(len(var)-1):
        print("Performing Cross Validation Fold: ",i+1)
        if i ==0:
            trainingSetArray = data_X[var[(i+1)]:], data_Y[var[(i+1)]:]
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            DualProblemSolution(trainingSetArray, testSetArray)
        elif i > 0 and i< 10:
            trainingSetArray = np.concatenate((data_X[var[(i+1)]:],data_X[:var[i]])), np.concatenate((data_Y[var[(i+1)]:],data_Y[:var[i]]))
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            DualProblemSolution(trainingSetArray,testSetArray)

# main method
if __name__ == '__main__':
    dataSet1 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set1.dat")
    dataSet2 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set2.dat")
    dataSet3 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set3.dat")
    dataSet4 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set4.dat")

    # Load the entire dataset
    entire_dataset = np.loadtxt(dataSet2)

    # Fetch the features and Target values
    data_X = np.asmatrix(entire_dataset[:,0:len(entire_dataset[0])-1])
    data_Y = np.asmatrix(entire_dataset[:,len(entire_dataset[0])-1]).T

    # Using the polynomial Features mapto Higher space
    polyNom = PolynomialFeatures(1)

    # Change the data_X to new Feature space
    data_X = polyNom.fit_transform(data_X)

    var = [x*(len(data_X)/10) for x in range(0,11)]
    crossValidation(var)

