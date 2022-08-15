import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import glob

import sys
from time import time

TRAINDIR = "trainData"
TESTDIR = "testData"

def main():
    #get data for cnn
    trainData, valData, testData = image_data(TRAINDIR, TESTDIR)
    #run cnn tests

    #vary number of nodes in dense layer, hold at 15 epochs
    cnn1 = train_cnn(trainData, valData, testData, 25, 15)
    cnn2 = train_cnn(trainData, valData, testData, 50, 15)
    cnn3 = train_cnn(trainData, valData, testData, 100, 15)
    cnn4 = train_cnn(trainData, valData, testData, 200, 15)

    #vary number of epochs holding at 75 dense layer nodes
    cnn5 = train_cnn(trainData, valData, testData, 75, 5)
    cnn6 = train_cnn(trainData, valData, testData, 75, 15)
    cnn7 = train_cnn(trainData, valData, testData, 75, 25)
    cnn8 = train_cnn(trainData, valData, testData, 75, 50)

    #Vary Filter shape a bit
    cnn9 = train_cnn(trainData, valData, testData, 50, 1, filterShape=(2,2))
    cnn10 = train_cnn(trainData, valData, testData, 50, 20, filterShape=(4,4))
    cnn11 = train_cnn(trainData, valData, testData, 50, 20, filterShape=(6,6))


    """
    #verify that the logging system works for cnn
    cnn12 = train_cnn(trainData, valData, testData, 5, 1)
    cnn13 = train_cnn(trainData, valData, testData, 10, 1)
    """

    # get data for dnn
    train_files, test_files = get_list_of_files(TRAINDIR, TESTDIR)
    np.random.shuffle(train_files)
    train_data, train_labels = read_data(train_files, scale=True)
    test_data, test_labels = read_data(test_files)
    enc = LabelEncoder().fit(train_labels)
    y_train = enc.transform(train_labels)
    y_train = to_categorical(y_train)
    y_test = enc.transform(test_labels)
    y_test = to_categorical(y_test)

    """
    #verify that the logging system works for dnn
    dnn1 = train_dnn(train_data, test_data, y_train, y_test, 10, 10, 1, enc, test_labels)
    dnn2 = train_dnn(train_data, test_data, y_train, y_test, 5, 5, 1, enc, test_labels)
    """


    #run dnn tests varying first dense layer size
    #second dense layer kept at 50
    dnn3 = train_dnn(train_data, test_data, y_train, y_test, 25, 50, 25, enc, test_labels)
    dnn4 = train_dnn(train_data, test_data, y_train, y_test, 50, 50, 25, enc, test_labels)
    dnn5 = train_dnn(train_data, test_data, y_train, y_test, 100, 50, 25, enc, test_labels)
    dnn6 = train_dnn(train_data, test_data, y_train, y_test, 200, 50, 25, enc, test_labels)

    #run dnn tests varying second dense layer size
    #first dense layer kept at 75
    dnn7 = train_dnn(train_data, test_data, y_train, y_test, 75, 10, 25, enc, test_labels)
    dnn8 = train_dnn(train_data, test_data, y_train, y_test, 75, 10, 25, enc, test_labels)
    dnn9 = train_dnn(train_data, test_data, y_train, y_test, 75, 10, 25, enc, test_labels)
    dnn10 = train_dnn(train_data, test_data, y_train, y_test, 75, 10, 25, enc, test_labels)

    #run dnn tests varying number of epochs
    #both dense layers kept at 75
    dnn11 = train_dnn(train_data, test_data, y_train, y_test, 75, 75, 5, enc, test_labels)
    dnn12 = train_dnn(train_data, test_data, y_train, y_test, 75, 75, 15, enc, test_labels)
    dnn13 = train_dnn(train_data, test_data, y_train, y_test, 75, 75, 25, enc, test_labels)
    dnn14 = train_dnn(train_data, test_data, y_train, y_test, 75, 75, 50, enc, test_labels)



def get_list_of_files(train_dir, test_dir):
    image_files = []
    train_images = glob.glob("{}/*/*.png".format(train_dir))
    test_images = glob.glob("{}/*/*.png".format(test_dir))

    return train_images, test_images

def image_to_vector(image: np.ndarray, scale: bool = False) -> np.ndarray:
    length, height, depth = image.shape

    if scale:
        image.reshape((1, length * height * depth))/255.0

    return image.reshape((1, length * height * depth))

def read_data(image_files, scale: bool = False):
    image_array = []
    labels = []
    for image_file in image_files:
        im = cv2.imread(image_file)
        im = image_to_vector(im, scale)
        image_array.append(im)
        #This line is problematic, on Mac/Linux change the split character to
        #"/"
        #On windows you need to change the split character to "\\"
        #which is just one backslash but the first is an escape character
        labels.append(image_file.split('\\')[-2])

    image_array = np.vstack(image_array)
    return image_array, labels

def image_data(train_dir, test_dir):

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        image_size=(256, 256),
        seed=1,
        batch_size=64,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=1,
        image_size=(256, 256),
        batch_size=64,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=1,
        image_size=(256, 256),
        batch_size=64,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb'
    )

    return train_data, val_data, test_data

def train_dnn(train_data, test_data, y_train, y_test, dense1, dense2, numEpoch, enc, test_labels):
    n_features = train_data.shape[1]

    model = Sequential()
    model.add(Dense(dense1, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    # model.add(Dropout(0.5))

    model.add(Dense(dense2, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    oldOut = sys.stdout
    start = time()
    newOut = open("dnn"+str(dense1)+"DenseOne"+str(dense2)+"DenseTwo"+str(numEpoch)+"Epochs.txt", 'w')
    sys.stdout = newOut
    print("------------------------------------------------------------------")
    print("Training Model")
    print("------------------------------------------------------------------")
    history = model.fit(train_data, y_train, epochs=numEpoch, batch_size=64, verbose=1, validation_split=0.3)
    print("------------------------------------------------------------------")
    print("Model Summary")
    print("------------------------------------------------------------------")
    model.summary()
    loss, acc = model.evaluate(test_data, y_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)
    print("------------------------------------------------------------------")
    print("Prediction Report, Test Set")
    print("------------------------------------------------------------------")
    y_predicted = model.predict(test_data)
    y_predicted = np.argmax(y_predicted, axis=-1)

    y_test = enc.transform(test_labels)
    y_test = to_categorical(y_test)

    y_test = np.argmax(y_test, axis=-1)
    print(classification_report(y_test, y_predicted, digits=4))

    print("------------------------------------------------------------------")
    print("Time Elapsed: "+str(time() - start)+" seconds")

    sys.stdout = oldOut

    # plot learning curves
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross Entropy')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.savefig("DNN"+str(dense1)+"DenseOne"+str(dense2)+"DenseTwo"+str(numEpoch)+"Epochs.png")
    pyplot.close()

    return model

def train_cnn(train_data, val_data, test_data, numDense, numEpochs, n_classes=5, filterShape=(3,3)):
    # define model
    model = Sequential()

    model.add(Conv2D(32, filterShape, activation='relu', kernel_initializer='he_uniform', input_shape=(256, 256, 3)))
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(64, filterShape, activation='relu', kernel_initializer='he_uniform', input_shape=(256, 256, 3)))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(numDense, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dropout(0.25))

    model.add(Dense(n_classes, activation='softmax'))

    # define loss and optimizer
    #sparse_categorical_crossentropy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    oldOut = sys.stdout
    if filterShape == (3, 3):
        newOut = open("CNN"+str(numDense)+"Dense"+str(numEpochs)+"Epochs.txt", 'w')
    else:
        newOut = open("CNN"+str(numDense)+"Dense"+str(numEpochs)+"Epochs"+str(filterShape[0])+str(filterShape[1])+"Filter.txt", 'w')
    sys.stdout = newOut
    start = time()
    print("------------------------------------------------------------------")
    print("Training Model")
    print("------------------------------------------------------------------")
    history = model.fit(train_data, epochs=numEpochs, validation_data=val_data, shuffle=True)

    print("------------------------------------------------------------------")
    print("Model Summary")
    print("------------------------------------------------------------------")
    model.summary()
    # evaluate the model
    loss, acc = model.evaluate(test_data, verbose=1)
    print('Accuracy: %.3f' % acc)

    print("------------------------------------------------------------------")
    print("Prediction Report, Validation Set")
    print("------------------------------------------------------------------")

    predict_cnn(val_data, model)
    print("------------------------------------------------------------------")
    print("Prediction Report, Test Set")
    print("------------------------------------------------------------------")
    predict_cnn(test_data, model)

    print("------------------------------------------------------------------")
    print("Time Elapsed: "+str(time() - start)+" seconds")
    sys.stdout = oldOut

    # plot learning curves
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross Entropy')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    if filterShape == (3, 3):
        pyplot.savefig("CNN"+str(numDense)+"Dense"+str(numEpochs)+"Epochs.png")
    else:
        pyplot.savefig("CNN"+str(numDense)+"Dense"+str(numEpochs)+"Epochs"+str(filterShape[0])+str(filterShape[1])+"Filter.png")
    pyplot.close()

    return model

def predict_cnn(test_img_data, model):
    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    for image_batch, label_batch in test_img_data:   # use dataset.unbatch() with repeat
       # append true labels
       y_true.append(label_batch)
       # compute predictions
       preds = model.predict(image_batch)
       # append predicted labels
       y_pred.append(np.argmax(preds, axis = - 1))

    # convert the true and predicted labels into tensors
    correct_labels = tf.concat([item for item in y_true], axis = 0)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)

    print(classification_report(np.argmax(correct_labels.numpy(), axis=1),
                                predicted_labels.numpy(),
                                target_names=test_img_data.class_names))

if __name__=="__main__":
    main()
