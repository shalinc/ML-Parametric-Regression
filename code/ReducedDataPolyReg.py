"""
    LINEAR REGRESSION WITH SINGLE FEATURE VARIABLE POLYNOMIAL MODEL

    This program is used to predict Y with a Given X value.
    It uses the POLYNOMIAL MODEL for TARGET PREDICTION
    There is 10 - FOLD CROSS VALIDATION APPLIED and MSE AND RSE are calculated for each FOLD
    This program also Implements a readymade  python function
    This method is used for comparison with our code
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from urllib.request import urlopen


# Method to perform Polynomial Model
def PerformPolyRegression(trainingSetArray, testSetArray, degree):

    # Get the Traning Feature & Training Targets
    X = trainingSetArray[0]
    Y = trainingSetArray[1]

    # Number of Samples
    m = len(trainingSetArray[0])
    print ("The number of samples (m) are:", m)

    # create a Matrix of ONES
    initOnesMatrix = np.ones((1,m))
    #print(initOnesMatrix)

    # Create Features of Polynomial Degree using List Comprehension
    poly_deg_feature = [np.power(X,i) for i in range (1,degree+1)]

    # Conversion into Matrix of particular size and shape
    poly_mat = np.asarray(poly_deg_feature, dtype=float)
    poly_mat = poly_mat.reshape(len(poly_mat),-1)

    # Add the poly features to exisiting features list
    Z = np.vstack((initOnesMatrix,poly_mat)).T
    #print (Z.shape, Y.shape)

    # Calculate THETA values - DOT PRODUCT(PSEUDO INVERSE (Z), Y)
    THETA = np.dot(np.linalg.pinv(Z),Y)
    print("The THETA: ", THETA.T)

    # Fit our own model on TEST DATA and find the TARGET Value
    X0 = 1
    yHat = []
    for xi in testSetArray[0]:
        calc = THETA[0].item(0) * X0
        for i in range(1,len(THETA)):
            calc += THETA[i].item(0) * xi.item(0)**i
        yHat.append(calc)

    print("MODEL TARGET VALUES:", yHat)

    # Fit our own model on TRAINING DATA and find the TARGET Value
    yHatTrain = []
    for xi in trainingSetArray[0]:
        calc = THETA[0].item(0) * X0
        for i in range(1,len(THETA)):
            calc += THETA[i].item(0) * xi.item(0)**i
        yHatTrain.append(calc)

    #print(yHatTrain)

    # Check with Built-in Python function
    #PR = PolynomialFeatures(degree)
    #PR.fit(X.reshape((m,1)), Y.reshape((m,1)))
    #print("THE BUILT IN PYTHON FUNCTION HAS THETA: ", PR.predict(testSetArray[0].reshape((len(testSetArray[0]),1))))

    # PLOTTING OF ENTIRE TRAINING DATA and TARGET OBTAINED
    """
    plt.title("ENTIRE DATA PLOT")
    plt.xticks()
    plt.yticks()
    plt.xlabel("X - FEATURE")
    plt.ylabel("y - TARGET")

    # Calculate the Indexes in sorted order to plot the graphs
    plotTrain = np.squeeze(np.asarray(trainingSetArray[0]))
    indexes_Train = plotTrain.argsort()
    plt.scatter(data_X, data_Y, color='blue')
    plt.scatter(plotTrain, np.asarray(yHatTrain),c='blue')
    #plt.plot(plotTrain[indexes_Train], np.asarray(yHatTrain)[indexes_Train],c='red')
    plt.show()
    """

    #PLOT the RESULTS OBTAINED

    # Calculate the Indexes in sorted order to plot the graphs
    plotTest = np.squeeze(np.asarray(testSetArray[0]))
    indexes = plotTest.argsort()
    plt.title("Cross Validation Results")
    plt.xticks()
    plt.yticks()
    plt.xlabel("X - FEATURE")
    plt.ylabel("y - TARGET")
    plt.scatter((testSetArray[0]), (testSetArray[1]))
    plt.plot(plotTest[indexes],np.asarray(yHat)[indexes], color='red')
    plt.show()

    # Calcualte the ERRORS
    calculate_errors(trainingSetArray, testSetArray, yHat, yHatTrain)

#Calculate RSE and MSE
def calculate_errors(trainingSetArray, testSetArray, yHat, yHatTrain):

    yHat = np.reshape(yHat,newshape=(len(yHat),1))
    yHatTrain = np.reshape(yHatTrain,newshape=(len(yHatTrain),1))

    # RSE TEST DATA
    RSE_TEST = calc_RSE(testSetArray, yHat)
    print("RSE FOR TEST DATA:",RSE_TEST)

    # RSE TRAIN DATA
    RSE_TRAIN = calc_RSE(trainingSetArray, yHatTrain)
    print("RSE FOR TRAIN DATA:",RSE_TRAIN)

    # MSE TEST DATA
    MSE_TEST = calc_MSE(testSetArray, yHat)
    print("MSE FOR TEST DATA:",MSE_TEST)

    # MSE TRAIN DATA
    MSE_TRAIN = calc_MSE(trainingSetArray, yHatTrain)
    print("MSE FOR TRAIN DATA:",MSE_TRAIN)

# Function to Calculate the RESIDUAL SUM OF ERRORS
def calc_RSE(Tarray,yHat):
    Mean_Test = np.mean(Tarray[1])
    numeratorRSE = np.power((np.subtract(yHat,Tarray[1])),2)
    denominatorRSE = np.power((np.subtract(Tarray[1],Mean_Test)),2)

    return (np.divide((np.sum(np.divide(numeratorRSE,denominatorRSE))),len(Tarray[1])))

# Function to Calculate MEAN SQUARE ERRORS
def calc_MSE(Tarray, yHat):
    numeratorRSE = np.power((np.subtract(yHat,Tarray[1])),2)
    return (np.divide((np.sum(numeratorRSE)),len(Tarray[1])))


# main method
if __name__ == '__main__':

     # All the dataset will be loaded at first for Single variate Linear regression
    dataSet1 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set1.dat")
    dataSet2 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set2.dat")
    dataSet3 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set3.dat")
    dataSet4 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set4.dat")

    # Input from Use the Degree of Polynomial
    degree = int(input("Enter the Degree of Polynomial (Integer values only): "))

    # Load the entire dataset
    entire_dataset = np.loadtxt(dataSet3)           # !!! CHANGE THE dataset parameter here to load different Datasets !!!

    #Split the DataSet into Features and y i.e. output/ target vector
    data_X = np.asmatrix(entire_dataset[:,0:len(entire_dataset[0])-1])      #FEATURE
    data_Y = np.asmatrix(entire_dataset[:,len(entire_dataset[0])-1]).T      #TARGET

    #Reduce the Data 30% and 50% of the original

    for i in [140, 100]:

        TrainingSetArray = data_X[i:], data_Y[i:]
        TestSetArray = data_X[i-50:i-30], data_Y[i-50:i-30]

        plt.title("Reduced samples")
        plt.xticks()
        plt.yticks()
        plt.xlabel("X")
        plt.ylabel("y")
        plt.scatter(data_X[i:], data_Y[i:],color='black')
        plt.show()

        PerformPolyRegression(TrainingSetArray, TestSetArray, degree)
