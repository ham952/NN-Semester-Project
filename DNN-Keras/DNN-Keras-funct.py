##########################################################################################################################
# Deep Neural Network for Image Classification using "signs dataset" from kaggle
# This is a Keras implementation of DNN by Hamza Rafique, during Masters Studies in IAA , Air University, Isb 2019 email : ham952@hotmail.com
# Keras functional API approach is used to implement DNN 
# Please feel free to use. Dont forget to acknowledge :)
##########################################################################################################################

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, Flatten
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

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)



start_time = time.time()


def DNN_Keras(input_shape=(64, 64, 3), classes=6):
    """
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)


    # output layer
    X = Flatten()(X_input)
    X1 = Dense(25, activation='relu', name = 'Layer-1')(X)
    X2 = Dense(12, activation='relu', name = 'Layer-2')(X1)
    X_out = Dense(classes, activation= 'softmax', name= 'fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X2)

    # Create model
    model = Model(inputs=X_input, outputs=X_out, name='DNN_Keras')

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
#---------------------------------------------------------------------
#                       Build Model
#---------------------------------------------------------------------

    model = DNN_Keras(input_shape = (64, 64, 3), classes = 6)
    #opt = SGD(lr=0.009,decay=1e-6, momentum=0.9, nesterov=True)
    #opt = SGD(lr=0.009)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

#---------------------------------------------------------------------
#                Perfrorm Training & Predictions
#---------------------------------------------------------------------
    history = model.fit(X_train, Y_train, epochs = 1000, batch_size = 64, validation_data = (X_test, Y_test))

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
