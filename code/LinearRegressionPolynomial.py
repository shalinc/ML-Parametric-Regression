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
import numpy.polynomial.polynomial as polyPlot
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

    # Check with Built-in Python function
    Coef = polyPlot.polyfit(np.ravel(X), np.ravel(Y),deg=degree)
    print("THE BUILT IN PYTHON FUNCTION HAS THETA: ",Coef)

    # Fit our own model on TEST DATA and find the TARGET Value
    X0 = 1
    yHat = []
    for xi in testSetArray[0]:
        calc = THETA[0].item(0) * X0
        for i in range(1,len(THETA)):
            calc += THETA[i].item(0) * xi.item(0)**i
        yHat.append(calc)

    print("Comparing OWN MODEL vs PYTHON MODEL of POLYNOMIAL MODEL")
    print("OWN MODEL TARGET VALUES:", yHat)

    # Fit PYTHON model on TEST DATA and find the TARGET Value
    X0 = 1
    yHatPython = []
    for xi in testSetArray[0]:
        calc = Coef[0].item(0) * X0
        for i in range(1,len(THETA)):
            calc += Coef[i].item(0) * xi.item(0)**i
        yHatPython.append(calc)

    print("PYTHON MODEL TARGET VALUES:", yHatPython)

    # Fit our own model on TRAINING DATA and find the TARGET Value
    yHatTrain = []
    for xi in trainingSetArray[0]:
        calc = THETA[0].item(0) * X0
        for i in range(1,len(THETA)):
            calc += THETA[i].item(0) * xi.item(0)**i
        yHatTrain.append(calc)

    #print(yHatTrain)

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

    plt.figure(1)

    plt.suptitle("OWN MODEL vs PYTHON MODEL (10 FOLD CV)", fontsize=14)
    plt.xticks()
    plt.yticks()
    plt.xlabel("X - FEATURE")
    plt.ylabel("y - TARGET")

    # PLOT ON TEST DATA - OWN MODEL
    ax = plt.subplot(211)
    ax.set_title("OWN MODEL", fontsize=10)
    plt.scatter((testSetArray[0]), (testSetArray[1]))
    plt.plot(plotTest[indexes],np.asarray(yHat)[indexes], color='red')

    # PLOT ON TEST DATA - PYTHON
    ax = plt.subplot(212)
    ax.set_title("PYTHON MODEL", fontsize=10)
    plt.scatter(testSetArray[0], testSetArray[1],color='black')
    plt.plot(plotTest[indexes],np.asarray(yHatPython)[indexes],color='blue')

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


# Perform 10- FOLD Cross validation
def crossValidation(var, degree):
    for i in range(len(var)-1):
        print("Performing Cross Validation Fold: ",i+1)
        if i ==0:
            trainingSetArray = data_X[var[(i+1)]:], data_Y[var[(i+1)]:]         # TRAINING SET having Feature Vector and Target Vector
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]     # TEST SET having Feature Vector and Target Vector
            PerformPolyRegression(trainingSetArray, testSetArray, degree)       # Do regression
        elif i > 0 and i< 10:
            trainingSetArray = np.concatenate((data_X[var[(i+1)]:],data_X[:var[i]])), np.concatenate((data_Y[var[(i+1)]:],data_Y[:var[i]]))
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            PerformPolyRegression(trainingSetArray,testSetArray, degree)

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

    # Plot the Original DataSet
    plt.title("Original DataSet")
    plt.xticks()
    plt.yticks()
    plt.scatter(data_X, data_Y,color='black')
    plt.show()

    # split into Training and Test Data for Cross Validation
    var = [x*(len(data_X)/10) for x in range(0,11)]         # create a Var list of Starting and Ending points of Folds
    crossValidation(var, degree)                            # Do Cross Validation

