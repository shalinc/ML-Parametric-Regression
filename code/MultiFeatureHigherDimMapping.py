"""
    MULTIVARIATE LINEAR REGRESSION MAPPING TO HIGHER DIMENSIONAL
    This program in intended to solve the multivariate regression problem
    Here I have mapped features to Higher Dimensional Space (N),
    so as to compare the results with normal feature space (n) Where, N > n
    The program accepts the Dimension to Map from the USER
    EX: Dimension: 2, if there are 2 (n) features, it will be mapped to 6 (N) features
    viz. 1, X1, X2, X1^2, X2^2, X1^X2
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from urllib.request import urlopen

# MULTI VARIATE MAPPING TO HIGHER DIMENSIONAL SPACE

def PerformMappingToHigherDimension(trainingSetArray, testSetArray):

    # Get the Traning Feature & Training Targets
    X = trainingSetArray[0]
    Y = trainingSetArray[1]

    # Number of Samples
    m = len(trainingSetArray[0])
    print ("The number of samples (m) are:", m)

    # Mapped data Z
    Z = X

    startTime = time.time()
    # Calculate THETA for High Dimension SPACE FEATURES
    THETA = np.dot(np.linalg.pinv(np.asarray(Z)),Y)
    print("The THETA: ",THETA.tolist())

    # Fit the Model to TEST DATA and Predict TARGET VAlUES
    X0 = 1
    yHat = []
    for xi in testSetArray[0]:
        calc = 0
        for i in range(0,len(THETA)):
            calc += THETA[i].item(0) * xi.item(i)
        yHat.append(calc)

    print("MODEL PREDICTED TARGET VALUES: ",yHat)

    # Fit the Model to TRAINING DATA and Predict TARGET VAlUES
    yHatTrain = []
    for xi in trainingSetArray[0]:
        calc = 0
        for i in range(0,len(THETA)):
            calc += THETA[i].item(0) * xi.item(i)
        yHatTrain.append(calc)

    print("Time taken for execution: ",time.time()-startTime," seconds")
    #print(yHatTrain)
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




# Performing 10- Fold cross validation
def crossValidation(var):
    for i in range(len(var)-1):
        print("Performing Cross Validation Fold: ",i+1)
        if i ==0:
            trainingSetArray = data_X[var[(i+1)]:], data_Y[var[(i+1)]:]
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            PerformMappingToHigherDimension(trainingSetArray, testSetArray)
        elif i > 0 and i< 10:
            trainingSetArray = np.concatenate((data_X[var[(i+1)]:],data_X[:var[i]])), np.concatenate((data_Y[var[(i+1)]:],data_Y[:var[i]]))
            testSetArray = data_X[var[i]:var[i+1]], data_Y[var[i]:var[i+1]]
            PerformMappingToHigherDimension(trainingSetArray,testSetArray)

# main method
if __name__ == '__main__':

    dataSet1 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set1.dat")
    dataSet2 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set2.dat")
    dataSet3 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set3.dat")
    dataSet4 = urlopen("http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set4.dat")
    dataSet5 = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data")
    dataSet6 = urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat")

    mapTO = int(input("Enter the Higher Dimension Space to Map Features to (Integer values only !!!): "))

    # Load the entire dataset
    entire_dataset = np.loadtxt(dataSet5)                   # !!! Change the dataset parameter here to change the dataset
    data_X = np.asmatrix(entire_dataset[:,0:len(entire_dataset[0])-1])
    data_Y = np.asmatrix(entire_dataset[:,len(entire_dataset[0])-1]).T

    print("The number of features originally were (n)", data_X.shape[1])
    # Using the polynomial Features mapto Higher space
    polyNom = PolynomialFeatures(mapTO)

    # Change the data_X to new Feature space
    data_X = polyNom.fit_transform(data_X)

    # The new feature
    print("Now at Higher Dimensional Feature Space is (N)", data_X.shape[1])

    if(data_X.shape[0] < data_X.shape[1]):
        print("Condition Check N > m ",data_X.shape[0]," (no. of samples) TARGET VALUES CAN BE WRONGLY INTERPRETED !!!")
        cont = input("Do you want to continue ? (Y/N)")
        if(str.upper(cont) == 'Y'):
            var = [x*(len(data_X)/10) for x in range(0,11)]
            crossValidation(var)
        else:
            print("PROGRAM IS STOPPING ...")
    else:
        var = [x*(len(data_X)/10) for x in range(0,11)]
        crossValidation(var)