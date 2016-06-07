"""
    LINEAR REGRESSION WITH SINGLE FEATURE VARIABLE

    This program is used to predict Y with a Given X value.
    It uses Single Variate Linear regression Technique to predict the value
    There is 10 - FOLD CROSS VALIDATION APPLIED and MSE AND RSE are calculated for each FOLD
    This program also Implements a readymade  python function to calculate Linear Regression
    This method is used for comparison with our code
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen


# Method to Perform Linear Regression
def PerformLinearRegression(trainingSetArray, testSetArray):

    # Get the Traning Feature & Training Targets
    X = trainingSetArray[0]
    Y = trainingSetArray[1]

    # Number of Samples
    m = len(trainingSetArray[0])
    print ("The number of samples (m) are:", m)

    # summation of X(i) and y(i)
    summation_Xi = (sum(trainingSetArray[0])).item(0)
    summation_yi = sum(trainingSetArray[1]).item(0)
    #print ("Sum of Xi and yi is",summation_Xi, summation_yi)

    # sum of square of X(i)
    sumOfSquare_Xi = sum(np.power(trainingSetArray[0],2)).item(0)
    #print ("The sum of square of Xi is",sumOfSquare_Xi)

    # sum of X(i)*y(i)
    sumOf_Xi_yi = np.sum(np.multiply(X, Y)).item(0)
    #print("The sum of Xi * yi is:", sumOf_Xi_yi)

    # matrix of Known values A
    A = np.matrix([[m, summation_Xi], [summation_Xi, sumOfSquare_Xi]])
    #print ("The know matrix A is:", A)

    # matrix of Known values B
    B = np.matrix([[summation_yi],[sumOf_Xi_yi]])
    #print ("The matrix B is:", B)

    # solve the Equation to get the values of THETA
    THETA = np.linalg.solve(A,B)
    print ("The THETA:", THETA)

    # Check with Built-in Python function
    LP = LinearRegression()
    LP.fit(X.reshape((m,1)), Y.reshape((m,1)))      #FIT DATA
    LinearRegression(copy_X=True, fit_intercept= True, n_jobs= 1, normalize=False)
    # PREDICT DATA
    yHatPython = LP.predict(testSetArray[0].reshape((len(testSetArray[0]),1)))

    print("THE BUILT IN PYTHON FUNCTION HAS THETA: ", LP.intercept_, LP.coef_)


    # Fit our own model on TEST DATA and find the TARGET Value
    yHat = []
    for xi in testSetArray[0]:
        yHat.append(THETA[0].item(0) + THETA[1].item(0) * xi.item(0))
    #print(yHat)

    # Fit PYTHON model on TEST DATA and find the TARGET Value
    #yHatPython = []
    #for xi in testSetArray[0]:
    #    yHatPython.append(LP.intercept_.item(0) + LP.coef_.item(0) * xi.item(0))

    print("Comparing OWN MODEL vs PYTHON MODEL of LINEAR REGRESSION")
    print("OWN MODEL TARGET VALUES:", yHat)
    print("PYTHON MODEL TARGET VALUES:", yHatPython.tolist())

    # Fit the model on TRAINING DATA and find the TARGET Value
    yHatTrain = []
    for xi in trainingSetArray[0]:
        yHatTrain.append(THETA[0].item(0) + THETA[1].item(0) * xi.item(0))

    # PLOTTING OF ENTIRE TRAINING DATA and TARGET OBTAINED
    """
    plt.title("ENTIRE DATA PLOT")
    plt.xticks()
    plt.yticks()
    plt.xlabel("X - FEATURE")
    plt.ylabel("y - TARGET")

    plt.scatter(trainingSetArray[0], trainingSetArray[1], c='black')
    plt.scatter(trainingSetArray[0], yHatTrain, c='red')
    plt.show()
    """

    # PLOT THE TEST RESULTS OBTAINED
    plt.figure(1)

    plt.suptitle("OWN MODEL vs PYTHON MODEL (10 FOLD CV)", fontsize=14)
    plt.xticks()
    plt.yticks()
    plt.xlabel("X - FEATURE")
    plt.ylabel("y - TARGET")

    # PLOT ON TEST DATA - OWN MODEL
    ax = plt.subplot(211)
    ax.set_title("OWN MODEL", fontsize=10)
    plt.scatter(testSetArray[0], testSetArray[1], color='black')
    plt.plot(testSetArray[0],yHat,color='red')

    # PLOT ON TEST DATA - PYTHON
    ax = plt.subplot(212)
    ax.set_title("PYTHON MODEL", fontsize=10)
    plt.scatter(testSetArray[0], testSetArray[1],color='black')
    plt.plot(testSetArray[0],yHatPython,color='blue')

    plt.show()

    # calculate the ERRORS
    calculate_errors(trainingSetArray, testSetArray, yHat, yHatTrain, yHatPython)


#Calculate the RSE and MSE
def calculate_errors(trainingSetArray, testSetArray, yHat, yHatTrain, yHatPython):

    # change in Dimension
    yHat = np.reshape(yHat,newshape=(len(yHat),1))
    yHatTrain = np.reshape(yHatTrain,newshape=(len(yHatTrain),1))
    yHatPython = np.reshape(yHatPython,newshape=(len(yHatPython),1))

    # RSE TEST DATA
    RSE_TEST = calc_RSE(testSetArray, yHat)
    print("RSE FOR TEST DATA (OWN MODEL):",RSE_TEST)

    # RSE TEST DATA PYTHON FUNCTION
    RSE_TEST = calc_RSE(testSetArray, yHatPython)
    print("RSE FOR TEST DATA (PYTHON MODEL):",RSE_TEST)

    # RSE TRAIN DATA
    RSE_TRAIN = calc_RSE(trainingSetArray, yHatTrain)
    print("RSE FOR TRAIN DATA:",RSE_TRAIN)

    # MSE TEST DATA
    MSE_TEST = calc_MSE(testSetArray, yHat)
    print("MSE FOR TEST DATA (OWN MODEL):",MSE_TEST)

    # MSE TEST DATA PYTHON FUNCTION
    MSE_TEST = calc_MSE(testSetArray, yHatPython)
    print("MSE FOR TEST DATA (PYTHON MODEL):",MSE_TEST)

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


# Performing 10- Fold cross validation
def crossValidation(var):

    for i in range(len(var)-1):
        print("Performing 10-Fold Cross Validation Fold: ",i+1)
        if i ==0:
            trainingSetArray = data_X[var[(i+1)]:], data_Y[var[(i+1)]:]         # TRAINING SET having Feature Vector and Target Vector
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]     # TEST SET having Feature Vector and Target Vector
            PerformLinearRegression(trainingSetArray, testSetArray)             # Do Linear regression
        elif i > 0 and i< 10:
            trainingSetArray = np.concatenate((data_X[var[(i+1)]:],data_X[:var[i]])), np.concatenate((data_Y[var[(i+1)]:],data_Y[:var[i]]))
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            PerformLinearRegression(trainingSetArray,testSetArray)

# main method
if __name__ == '__main__':

    # All the dataset will be loaded at first for Single variate Linear regression
    dataSet1 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set1.dat")
    dataSet2 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set2.dat")
    dataSet3 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set3.dat")
    dataSet4 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set4.dat")

    # Load the entire dataset
    entire_dataset = np.loadtxt(dataSet1)           # !!! CHANGE THE dataset parameter here to load different Datasets !!!

    #Split the DataSet into Features and y i.e. output/ target vector
    data_X = np.asmatrix(entire_dataset[:,0:len(entire_dataset[0])-1])      #FEATURE
    data_Y = np.asmatrix(entire_dataset[:,len(entire_dataset[0])-1]).T      #TARGET

    # Plot the Original DataSet
    plt.title("Original DataSet")
    plt.xticks()
    plt.yticks()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.scatter(data_X, data_Y,color='black')
    plt.show()

    # split into Training and Test Data for Cross Validation
    var = [x*(len(data_X)/10) for x in range(0,11)]         #create a Var list of Starting and Ending points of Folds
    crossValidation(var)                                    # Do Cross Validation
