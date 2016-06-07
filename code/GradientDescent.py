"""
    ITERATIVE SOLUTION USING STOCHASTIC GRADIENT DESCENT ALGORITHM
    This program is intended to solve the Regression Problem using Iterative form
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as skl
from urllib.request import urlopen
from sklearn.preprocessing import PolynomialFeatures

# ITERATIVE METHODOLOY - STOCHASTIC GRADIENT DESCENT

def PerformGradientDescent(trainingSetArray, testSetArray):

    # Convert array to matrix
    X = trainingSetArray[0]
    Y = trainingSetArray[1]

    # Number of Samples
    m = len(X)
    print ("The number of samples are:", m)

    Z = X

    #Intiliaze the THETA MATRIX for initial prediction
    THETA = np.ones((Z.shape[1]))

    J_THETA_HISTORY = [] # store the previous COST FUNCTION

    numOfIterations = 70    # Number of Iterations
    ALPHA = 0.0005          # APLHA LEARNING RATE


    # Peform Iterative Gradient Descent
    for i in range(numOfIterations):
        Gradient = np.zeros(len(THETA))             # Initialize gradient to ZEROS
        errors = []                                 # Initialize Empty ERRORS List
        for j in range(len(Z)):                     # Perform for all SAMPLE POINTS
            error = np.dot(Z[j], THETA.T) - Y[j]    # CALCULATE ERROR -> (Z(dot)THETA.TRANSPOSE) - Y[j]
            errors.append(np.power(error,2))
            #print (error)
            temp_Gradient = np.asarray(Z[j]) * error.tolist()   # Calculate Gradient
            Gradient = Gradient + temp_Gradient

        THETA = THETA - (ALPHA * Gradient)          # Calculate the new THETA from Previous THETA value
        J_THETA = np.sum(errors)/(2*m)              # J(THETA) i.e. COST FUNCTION
        J_THETA_HISTORY.append(J_THETA)             # STORE ALL THE COST FUNCTIONS EVERY ITERATION

    print("The THETA: ", THETA)
    print("COST FUNCTION FOR ALL ITERATIONS", J_THETA_HISTORY)

    """
    plt.title("COST FUNCTION")
    plt.xticks(); plt.yticks()
    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    plt.plot([i for i in range(numOfIterations)], J_THETA_HISTORY, linewidth=2.0)
    plt.show()
    """

    # FIT and PREDICT ON TEST DATA
    yHat = []
    X0 = 1
    for xi in testSetArray[0]:
        calc = 0
        for i in range(0,len(THETA[0])):
            calc += THETA[0].item(i) * xi.item(i)
        yHat.append(calc)

    print("MODEL PREDICTED TARGET VALUES: ",yHat)

    # FIT and PREDICT ON TEST DATA
    yHatTrain = []
    # Fit the model on TRAINING DATA
    for xi in trainingSetArray[0]:
        calc = 0
        for i in range(0,len(THETA[0])):
            calc += THETA[0].item(i) * xi.item(i)
        yHatTrain.append(calc)

    #print(yHatTrain)

    # Call the calculate errors function to find the MSE
    calculate_errors(trainingSetArray, testSetArray, yHat, yHatTrain)


#Calculate RSE and MSE
def calculate_errors(trainingSetArray, testSetArray, yHat, yHatTrain):

    yHat = np.reshape(yHat,newshape=(len(yHat),1))
    yHatTrain = np.reshape(yHatTrain,newshape=(len(yHatTrain),1))

    # MSE TEST DATA
    MSE_TEST = calc_MSE(testSetArray, yHat)
    print("MSE FOR TEST DATA:",MSE_TEST)

    # MSE TRAIN DATA
    MSE_TRAIN = calc_MSE(trainingSetArray, yHatTrain)
    print("MSE FOR TRAIN DATA:",MSE_TRAIN)


def calc_RSE(Tarray,yHat):
    Mean_Test = np.mean(Tarray[1])
    numeratorRSE = np.power((np.subtract(yHat,Tarray[1])),2)
    denominatorRSE = np.power((np.subtract(Tarray[1],Mean_Test)),2)

    return (np.divide((np.sum(np.divide(numeratorRSE,denominatorRSE))),len(Tarray[1])))

# Calculate the MSE
def calc_MSE(Tarray, yHat):
    numeratorRSE = np.power((np.subtract(yHat,Tarray[1])),2)
    return (np.divide((np.sum(numeratorRSE)),len(Tarray[1])))


# Perform 10- FOLD cross Validation
def crossValidation(var):
    for i in range(len(var)-1):
        print("Performing Cross Validation Fold: ",i+1)
        if i ==0:
            trainingSetArray = data_X[var[(i+1)]:], data_Y[var[(i+1)]:]
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            PerformGradientDescent(trainingSetArray, testSetArray)
        elif i > 0 and i< 10:
            trainingSetArray = np.concatenate((data_X[var[(i+1)]:],data_X[:var[i]])), np.concatenate((data_Y[var[(i+1)]:],data_Y[:var[i]]))
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            PerformGradientDescent(trainingSetArray,testSetArray)

# main method
if __name__ == '__main__':

    dataSet1 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set1.dat")
    dataSet2 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set2.dat")
    dataSet3 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set3.dat")
    dataSet4 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set4.dat")

    #mapTO = int(input("Enter the Higher Dimension Space to Map Features to (Integer values only !!!): "))

    # Load the entire dataset
    entire_dataset = np.loadtxt(dataSet1)                                   # !!! Change the dataset parameter Here
    data_X = np.asmatrix(entire_dataset[:,0:len(entire_dataset[0])-1])
    data_Y = np.asmatrix(entire_dataset[:,len(entire_dataset[0])-1]).T

    #Mapping to Higher Dimensional Feature space
    polyNom = PolynomialFeatures(1)
    data_X = polyNom.fit_transform(data_X)

    # Perform 10- Fold validation
    var = [x*(len(data_X)/10) for x in range(0,11)]
    crossValidation(var)

