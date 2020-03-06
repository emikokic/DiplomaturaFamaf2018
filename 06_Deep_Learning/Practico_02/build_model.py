#!/users/mferreyra/.virtualenvs/diplodatos-dl/bin/python

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.backend.tensorflow_backend import set_session

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def limit_memory(val):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = val
    set_session(tf.Session(config=config))


def read_args():
    parser = argparse.ArgumentParser(description='Exercise 1')

    """
    Here you have some examples of classifier parameters.
    You can add more arguments or change these if you need to.
    """
    parser.add_argument('--num_units',
                        default=512,
                        type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Number of instances in each batch.')
    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='Number of training epochs.')
    parser.add_argument('--experiment_name',
                        default=None,
                        type=str,
                        help='Name of the experiment, used in the filename'
                             'where the results are stored.')
    args = parser.parse_args()

    return args


def load_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    np.random.seed(1)  # For Reproducibility

    limit_memory(0.1)
    args = read_args()

    # Load CIFAR-10 Dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # Parameters
    NUM_CLASSES = 10
    INPUT_SIZE = 32 * 32 * 3
    TRAIN_EXAMPLES = 50000
    TEST_EXAMPLES = 10000

    # Reshape the dataset to convert the examples from 2D matrixes to 1D arrays
    X_train = X_train.reshape(TRAIN_EXAMPLES, INPUT_SIZE)
    X_test = X_test.reshape(TEST_EXAMPLES, INPUT_SIZE)

    # Normalize the input
    X_train = X_train / 255  # RGB
    X_test = X_test / 255

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Necesitamos redimensionar el input para obtener la imagen en 2D
    IMG_ROWS = 32  # Heigth
    IMG_COLS = 32  # Width
    CHANNELS = 3   # Color Channels (RGB)

    X_train = X_train.reshape(X_train.shape[0], IMG_ROWS, IMG_COLS, CHANNELS)
    X_test = X_test.reshape(X_test.shape[0], IMG_ROWS, IMG_COLS, CHANNELS)

    # TODO 3: Build the Keras model
    model = Sequential()
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        input_shape=X_train.shape[1:],
        activation='relu'
    ))
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(args.num_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )

    # TODO 4: Fit the model
    history = model.fit(
                x=X_train,
                y=y_train,
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=1,
                validation_split=0.1
    )

    """
    TODO 5: Evaluate the model, calculating the metrics.

    Option 1: Use the model.evaluate() method. For this, the model must be
              already compiled with the metrics.
    """
    performance = model.evaluate(X_test, y_test)

    # TODO 6: Save the results.
    model.save('model_{}.h5'.format(args.experiment_name))

    # One way to store the predictions:
    print("=" * 80)
    print("Batch Size = {}".format(args.batch_size))
    print("Epochs = {}".format(args.epochs))
    print("-" * 80)
    print("History = {}".format(history.history))
    print("-" * 80)
    print("Test loss     = {}".format(performance[0]))
    print("Test accuracy = {}".format(performance[1]))
    print("=" * 80)

    # results = pd.DataFrame(performance)
    # results.loc[:, 'true_label'] = y_test
    # results.to_csv('predicitions_{}.csv'.format(args.experiment_name),
    #                index=False)
