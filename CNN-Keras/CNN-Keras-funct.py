##########################################################################################################################
# Convolutional Neural Network (CNN) for Image Classification using "signs dataset" from kaggle
# This is a Keras implementation of CNN by Hamza Rafique, IAA , Air University, Isb email : ham952@hotmail.com
# Keras functional API approach is used to implement CNN 
# Please feel free to use. Dont forget to acknowledge :)
##########################################################################################################################

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, Flatten,ZeroPadding2D,MaxPooling2D,Conv2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import time
import os
from Utils import *
from keras.utils import to_categorical
from keras.optimizers import SGD
import argparse
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)



start_time = time.time()


def CNN_Keras(input_shape=(64, 64, 3), classes=6):
    """
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D(padding=(3,3))(X_input)

    # CONV -> RELU -> MaxPool Block applied to X
    X = Conv2D(32, (4, 4), strides=(1, 1), name='conv0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((8, 8),strides=(8, 8), name='max_pool0')(X)
    

    # CONV -> RELU -> MaxPool Block applied to X
    X = Conv2D(64, (2, 2), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4),strides=(8, 8), name='max_pool1')(X)

    
    # output layer
    X = Flatten()(X)
    X_out = Dense(classes, activation= 'softmax', name= 'fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X_out, name='CNN_Keras')

    return model

#---------------------------------------------------------------------
#                       Load Data
#---------------------------------------------------------------------
# Load Dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
#Y_train = convert_to_one_hot(Y_train_orig, 6).T
#Y_test = convert_to_one_hot(Y_test_orig, 6).T

Y_train = to_categorical(Y_train_orig,6)
Y_train = np.squeeze(Y_train )

Y_test = to_categorical(Y_test_orig,6)
Y_test = np.squeeze(Y_test)

print ("Y_train original: " + str(Y_train_orig.shape))
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def main():

    parser = argparse.ArgumentParser(description='Constructing CNN for classification')
    parser.add_argument('--epochs',
                        required = False,
                        default = 212,
                        type=int,
                        metavar='212',
                        help="number of epochs to run")
    args = parser.parse_args()
#---------------------------------------------------------------------
#                       Build Model
#---------------------------------------------------------------------

    model = CNN_Keras(input_shape = (64, 64, 3), classes = 6)
    #opt = SGD(lr=0.009,decay=1e-6, momentum=0.9, nesterov=True)
    #opt = SGD(lr=0.009)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

#---------------------------------------------------------------------
#                Perfrorm Training & Predictions
#---------------------------------------------------------------------
    history = model.fit(X_train, Y_train, epochs = args.epochs, batch_size = 64, validation_data = (X_test, Y_test))

    preds = model.evaluate(X_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.figure()

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')

#---------------------------------------------------------------------
#                Model Summary & saving
#---------------------------------------------------------------------

    print("Execution took : %s mins " % ((time.time() - start_time)/60))

    plt.show()

if __name__ == '__main__':
    main()
