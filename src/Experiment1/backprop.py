"""
mnistNN.py -- Sean Sweeney 26 April, 2022
A class which implements a two layer neural network for classifying handwritten
digits from the  MNIST data set.
"""

import numpy as np
import sys
from statistics import mean
import glob
import cv2
import matplotlib.pyplot as plot
from time import time

# change amount of array that gets printed, for error checking
#np.set_printoptions(threshold=sys.maxsize)

def main():
    nn = dnn(10, 0.1, 0.9)
    runTest(nn, 100, 0.1, 0.9, 30, "Hand Coded Back Propegation")


"""
Sigmoid is used as the activation function - ie for "squashing" the data
    -chose not to make this a class member
"""
def sig(x):
    return 1/(1+np.exp(-x))


"""
A class that implements a two layer neural network for classifying handwritten
digits from the MNIST data set
Three Hyperparameters to define each instance of the class
    -hid: number of nodes in the hidden layer, must be at least 1
    -eta: learning rate, a value between 0 and 1
    -alpha: momentum, a value between 0 and 1
Data from the mnist_train.csv and mnist_test.csv files is automatically
loaded in for each instance created.
"""
class dnn(object):


    def __init__(self, i_hid, i_eta, i_alpha):
        # sizes do not account for bias units
        self.inSize = 256*256*3
        self.outSize = 10
        # hyperparameters
        self.hidSize = i_hid
        self.eta = i_eta
        self.alpha = i_alpha
        # +1 to account for bias unit in input and hidden layer
        self.hidW = (0.1)*(np.random.rand(self.inSize+1, self.hidSize)-0.5)
        self.outW = (0.1)*(np.random.rand(self.hidSize+1, self.outSize)-0.5)
        # data to work with
        self.testData, self.testLabels, self.trainData, self.trainLabels = self.getImageData("trainData", "testData")
        # can use activation function on entire numpy array at once
        self.act = np.vectorize(sig)
        # get size of data sets up front
        self.testSize = len(self.testData)
        self.trainSize = len(self.trainData)


    """
    Resets nn to fresh starting state with new hyper parameters
    Useful so that we don't have to load data every time
    """
    def reset(self, new_hid, new_eta, new_alpha):
        #reset hyperparameters
        self.hidSize = new_hid
        self.eta = new_eta
        self.alpha = new_alpha
        #reset weights
        self.hidW = (0.1)*(np.random.rand(self.inSize+1, self.hidSize)-0.5)
        self.outW = (0.1)*(np.random.rand(self.hidSize+1, self.outSize)-0.5)

    """
    Gets image data into flattened arrays
    Returns normalized, flattened arrays for each test, train data set
    """
    def getImageData(self, trainDir, testDir):
        trainImages, testImages = self.get_list_of_files(trainDir, testDir)
        trainData, trainLabels = self.read_data(trainImages)
        trainData *= (1/255)
        testData, testLabels = self.read_data(testImages)
        testData *= (1/255)
        return testData, testLabels, trainData, trainLabels


    """
    Sets the training data to be a fraction of the training data as per the
    frac argument
    Parameter is the denominator of the fraction, i.e. if you want half the data
    frac=2, if you want a quarter frac=4
    """
    def setFracTrain(self, frac):
        freq = [0 for _ in range(self.outSize)]
        newData = []
        i = 0
        numLoops = 0
        np.random.shuffle(self.trainData)
        while len(newData) < (self.trainSize//frac) and numLoops < 5:
            curVec = self.trainData[i]
            label = int(curVec[0])
            if mean(freq)+3 > freq[label]:
                newData.append(curVec)
                freq[label] += 1
            i += 1
            if i >= self.trainSize:
                i = 0
                numLoops += 1
                np.random.shuffle(self.trainData)
        self.trainData = np.array(newData)
        self.trainSize = len(self.trainData)


    """
    Stolen wholesale from a slack overflow question (not even an answer!)
    ://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unisone
    It shuffles two arrays in the same order so that the labels and the data can be
    shuffled but still stay the same relative to eachother
    """
    def shuffle_in_unison(self, a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    """
    Run a single epoch of training
    (1) Shuffle input so that it doesn't come in the same order each epoch
    (2) Forward propogate from input to hidden layer
    (3) Forward propogate from hidden layer to output
    (4) Get expected output from label (first element of each training data)
    (5) Backpropogate...
    """
    def runEpoch(self):
        # so we can return the accuracy of each epoch
        numCorrect = 0
        # randomize input order
        self.trainData, self.trainLabels = self.shuffle_in_unison(self.trainData, self.trainLabels)
        # to hold on to the last weight change for momentum
        oldDelOutW = np.zeros(self.outW.shape)
        oldDelHidW = np.zeros(self.hidW.shape)
        #counter
        i = 0
        for inData in self.trainData:
            # get the label
            label = self.trainLabels[i]
            # put in bias unit
            inData = np.insert(inData, 0, 1)
            # forward propogate to hidden layer
            hidOut = self.act(np.dot(inData, self.hidW))
            # put in bias unit
            hidOut = np.insert(hidOut, 0, 1)
            # forward propogate to output layer
            out = self.act(np.dot(hidOut, self.outW))
            # get target vecot
            t = self.getTVec(label)
            # calculate error terms
            outError = out*(1 - out)*(t - out)
            # note that we need to disclude bias weights from output vector
            hidError = hidOut[1:]*(1-hidOut[1:])*np.dot(self.outW[1:,:], outError)
            # calculate weight changes
            # need to take the transpose of the outputs to get a matrix
            delOutW = self.eta * outError * np.atleast_2d(hidOut).T
            delHidW = self.eta * hidError * np.atleast_2d(inData).T
            # update weights
            self.outW += delOutW + (self.alpha*oldDelOutW)
            self.hidW += delHidW + (self.alpha*oldDelHidW)
            # reset the old delW for use in momentum calc
            oldDelOutW = delOutW
            oldDelHidW = delHidW
            # if prediction was correct increment
            if np.argmax(out) == label:
                numCorrect += 1
            #get the bias unit out of the data
            inData = np.delete(inData, 0)
            #update counter
            i += 1
        #return the accuracy of the training epoch
        return numCorrect/self.trainSize


    """
    Takes an integer label, returns a nparray of length self.outSize with 0.1
    at every index except for the index corrosponding to label, which is 0.9
    """
    def getTVec(self, label):
        tVec = (0.1)*(np.ones(self.outSize))
        tVec[label] = 0.9
        return tVec


    """
    Train the NN for numEpoch epochs and return an array with accuracy of each
    """
    def runTraining(self, numEpoch):
        # array to hold accuracy of each epoch to be returned
        accTrain = []
        accTest = []
        for _ in range(numEpoch):
            #run epoch and store accuracy
            trainAcc = self.runEpoch()
            accTrain.append(trainAcc)
            testAcc = self.getTestAccuracy()
            accTest.append(testAcc)
        return accTrain, accTest


    """
    Runs all the test data on the current state of the NN and returns the
    accuracy
    """
    def getTestAccuracy(self):
        numCorrect = 0
        i = 0
        for inData in self.testData:
            label = self.testLabels[i]
            inData = np.insert(inData, 0, 1)
            hidOut = self.act(np.dot(inData, self.hidW))
            hidOut = np.insert(hidOut, 0, 1)
            out = self.act(np.dot(hidOut, self.outW))
            inData = np.delete(inData, 0)
            if np.argmax(out) == label:
                numCorrect += 1
            i += 1
        return numCorrect/self.testSize


    """
    Returns a two-dimensional array to represent a confusion matrix
    """
    def getConfusionMatrix(self):
        confusion = [[0 for _ in range(10)] for _ in range(10)]
        i = 0
        for inData in self.testData:
            label = self.testLabels[i]
            inData = np.insert(inData, 0, 1)
            hidOut = self.act(np.dot(inData, self.hidW))
            hidOut = np.insert(hidOut, 0, 1)
            out = self.act(np.dot(hidOut, self.outW))
            prediction = np.argmax(out)
            confusion[label][prediction] += 1
            inData[0] = label
            i += 1
        return confusion

    """
    Gets lists of files and associated edges, for use in read_data
    """
    def get_list_of_files(self, train_dir, test_dir):
        image_files = []
        train_images = glob.glob("{}/*/*.png".format(train_dir))
        test_images = glob.glob("{}/*/*.png".format(test_dir))
        return train_images, test_images

    """
    turns an image array into a flattened numpy array
    """
    def image_to_vector(self, image: np.ndarray) -> np.ndarray:
        length, height, depth = image.shape
        return image.reshape((1, length * height * depth))

    """
    Takes a list of image files and returns a data set of those as
    nparray vector
    """
    def read_data(self, image_files):
        image_array = []
        labels = []
        for image_file in image_files:
            im = cv2.imread(image_file)
            im = self.image_to_vector(im)
            image_array.append(im)
            labelStr = image_file.split('\\')[-2]
            if labelStr == "healthy":
                labels.append(0)
            elif labelStr == "mild":
                labels.append(1)
            elif labelStr == "moderate":
                labels.append(2)
            elif labelStr == "severe":
                labels.append(3)
            elif labelStr == "proliferate":
                labels.append(4)
            #image_array = np.vstack(image_array)
        image_array = np.array(image_array, dtype=np.float64)
        labels = np.array(labels)
        return image_array, labels

"""
Takes a neural network object, three hyperparameters for the object, a number
of epochs to run and a string to name the test output
Resets the neural network to have the specified hyperparameters and
re-randomizes the weights, then trains the network for the specified number
of epochs
Produces three files:
    (1) A plot of the accuracy on the training data during each epoch
    (2) A plot of the accuracy on the testing data after each epoch
    (3) A text file of the confusion matrix on test set after training
"""
def runTest(nn, hidSize, eta, alpha, numEpoch, testName):
    nn.reset(hidSize, eta, alpha)
    data = nn.runTraining(numEpoch)
    trainData = data[0]
    testData = data[1]
    makeChart(trainData, "Training Accuracy", testName+" Train")
    makeChart(testData, "Test Accuracy", testName+" Test")
    cMat = nn.getConfusionMatrix()
    makeConfusionMatrix(cMat, testName)


"""
Function used to make the plots of accuracy per epoch, relies on pyplot,
makes a .png figure in the current directory
"""
def makeChart(dataList, yLabel, title):
    n = len(dataList)
    xAx = list((i for i in range(n)))
    plot.scatter(xAx, dataList, s=1)
    plot.xlabel("Epoch")
    plot.ylabel(yLabel)
    plot.title(title)
    plot.xlim([0, n+(n//10)])
    plot.ylim([(9/10)*(min(dataList)), (11/10)*(max(dataList))])
    plot.savefig(title.replace(" " ,"").replace(",", "")+".png")
    plot.close()


"""
Takes a multidimensional python array such as is produced by the
mnistNN.getCOnfusionMatrix function and produces a well formated .txt file
containing that matrix
"""
def makeConfusionMatrix(matrix, testType):
    oldOut = sys.stdout
    newOut = open((testType+"Confusion.txt").replace(" ",""), 'w')
    sys.stdout = newOut
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
    sys.stdout = oldOut

if __name__ == "__main__":
    main()
