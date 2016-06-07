"""
    MULTIVARIATE LINEAR REGRESSION
    This program in intended to solve the multivariate regression problem
    Here I haven't mapped it to Higher Dimensional Space, so as to compare the results with higher dimensional
    There is another program which does the MAPPING to Higher Dimensional space
    Here we have a 3-D plot but is limited when number of features are 2, so that it can be plotted in 3-D
    For others i.e. more than 2 features no-plots are plotted
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from urllib.request import urlopen


# MULTIVARIATE REGRESSION NORMAL SPACE OF 'n' Features

def PerformMultiVarRegression(trainingSetArray, testSetArray):

    # Convert array to matrix
    X = trainingSetArray[0]
    Y = trainingSetArray[1]

    # Number of Samples
    m = len(trainingSetArray[0])
    print ("The number of samples (m) are:", m)

    # create a Matrix of ONES
    initOnesMatrix = np.ones((m,1))
    #print(initOnesMatrix)

    # Append the Features to matrix of ones, forming a complete feature data
    Z = np.c_[initOnesMatrix, X]
    #print(Z)

    # Calculate THETA
    THETA = np.dot(np.linalg.pinv(Z),Y)
    print("The THETA: ",THETA.tolist())

    # Fit the MODEL and Predict the TARGET on TEST DATA SET
    X0 = 1
    yHat = []
    for xi in testSetArray[0]:
        calc = THETA[0].item(0) * X0
        for i in range(1,len(THETA)):
            calc += THETA[i].item(0) * xi.item(i-1)
        yHat.append(calc)

    print("MODEL PREDICTED TARGET VALUES: ",yHat)

    # Fit the MODEL and Predict the TARGET on TEST DATA SET
    yHatTrain = []
    for xi in trainingSetArray[0]:
        calc = THETA[0].item(0) * X0
        for i in range(1,len(THETA)):
            calc += THETA[i].item(0) * xi.item(i-1)
        yHatTrain.append(calc)

    #print(yHatTrain)

    #PLOT THE 3-D Data only if features are 2

    if data_X.shape[1] == 2:
        plotTest = np.squeeze(np.asarray(testSetArray[0]))
        indexes = plotTest.argsort()

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        plotX = [testSetArray[0].item(i*2) for i in range(int(len(testSetArray[0])))]
        plotY = [testSetArray[0].item((i*2)+1) for i in range(int(len(testSetArray[0])))]
        plotZ = testSetArray[1]

        ax.scatter(np.asarray(plotX), np.asarray(plotY), np.asarray(plotZ), label='Test Value')
        ax.scatter(np.asarray(plotX), np.asarray(plotY), np.asarray(yHat),color='red',marker='^', label='Predicted Value')
        ax.legend()

        plt.title("Cross Validation Results")
        plt.xticks()
        plt.yticks()
        plt.xlabel("X1 - FEATURE")
        plt.ylabel("X2 - FEATURE")
        ax.set_zlabel("y - TARGET")
        plt.show()
        #print(skl.mean_squared_error(testSetArray[1],yHat))
    calculate_errors(trainingSetArray, testSetArray, yHat, yHatTrain)

#Calculate RSE and MSE
def calculate_errors(trainingSetArray, testSetArray, yHat, yHatTrain):

    yHat = np.reshape(yHat,newshape=(len(yHat),1))
    yHatTrain = np.reshape(yHatTrain,newshape=(len(yHatTrain),1))

    # RSE TEST DATA
    #RSE_TEST = calc_RSE(testSetArray, yHat)
    #print("RSE FOR TEST DATA:",RSE_TEST)

    # RSE TRAIN DATA
    #RSE_TRAIN = calc_RSE(trainingSetArray, yHatTrain)
    #print("RSE FOR TRAIN DATA:",RSE_TRAIN)

    # MSE TEST DATA
    MSE_TEST = calc_MSE(testSetArray, yHat)
    print("MSE FOR TEST DATA:",MSE_TEST)

    # MSE TRAIN DATA
    MSE_TRAIN = calc_MSE(trainingSetArray, yHatTrain)
    print("MSE FOR TRAIN DATA:",MSE_TRAIN)


# Calculate RESIDUAL SQUARED ERROR
def calc_RSE(Tarray,yHat):
    Mean_Test = np.mean(Tarray[1])
    numeratorRSE = np.power((np.subtract(yHat,Tarray[1])),2)
    denominatorRSE = np.power((np.subtract(Tarray[1],Mean_Test)),2)

    return (np.divide((np.sum(np.divide(numeratorRSE,denominatorRSE))),len(Tarray[1])))

# Calculate MEAN SQUARED ERROR
def calc_MSE(Tarray, yHat):
    numeratorMSE = np.power((np.subtract(yHat,Tarray[1])),2)
    return (np.divide((np.sum(numeratorMSE)),len(Tarray[1])))


# Perform 10-Fold Cross Validation
def crossValidation(var):
    for i in range(len(var)-1):
        print("Performing Cross Validation Fold: ",i+1)
        if i ==0:
            trainingSetArray = data_X[var[(i+1)]:], data_Y[var[(i+1)]:]
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            PerformMultiVarRegression(trainingSetArray, testSetArray)
        elif i > 0 and i< 10:
            trainingSetArray = np.concatenate((data_X[var[(i+1)]:],data_X[:var[i]])), np.concatenate((data_Y[var[(i+1)]:],data_Y[:var[i]]))
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            PerformMultiVarRegression(trainingSetArray,testSetArray)

# main method
if __name__ == '__main__':

    dataSet1 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set1.dat")
    dataSet2 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set2.dat")
    dataSet3 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set3.dat")
    dataSet4 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set4.dat")
    dataSet5 = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data")
    dataSet6 = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat")


    # Load the entire dataset
    entire_dataset = np.loadtxt(dataSet6)                                   # !!! Change the dataset parameter to change the dataset
    data_X = np.asmatrix(entire_dataset[:,0:len(entire_dataset[0])-1])
    data_Y = np.asmatrix(entire_dataset[:,len(entire_dataset[0])-1]).T

    # Plot the Original DataSet
    if (data_X.shape[1] == 2):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter(np.asarray(data_X[:,0]),np.asarray(data_X[:,1]), np.asarray(data_Y),label="orginal data")
        ax.legend()
        plt.title("Original DataSet")
        plt.xticks()
        plt.yticks()

        plt.show()

    var = [x*(len(data_X)/10) for x in range(0,11)]
    crossValidation(var)
