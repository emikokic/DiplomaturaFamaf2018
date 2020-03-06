#!/users/mferreyra/.virtualenvs/diplodatos-cv/bin/python3.6

# https://keras.io/applications/#documentation-for-individual-models
import os
from keras.applications.mobilenet import MobileNet
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import pandas as pd
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def read_args():
    parser = argparse.ArgumentParser(
        description='Build Model 1',
        formatter_class=argparse.RawTextHelpFormatter
    )

    """
    Here you have some examples of classifier parameters.
    You can add more arguments or change these if you need to.
    """

    parser.add_argument(
        '--epochs',
        # default=10,
        type=int,
        help="Number of training epochs."
    )

    parser.add_argument(
        '--model_type',
        # default=0,
        type=str,
        choices=['1', '2'],
        required=True,
        help="Elige una de las arquitecturas de la NN"
    )

    parser.add_argument(
        '--train_base_layers',
        # default=0,
        type=int,
        choices=[0, 1],
        required=True,
        help="""\
Entrenar o no las capas anteriores a la clasificacion de MobileNet\
""")

    parser.add_argument(
        '--drop_prob',
        # default=0.5,
        type=float,
        choices=[0.0, 0.25, 0.5, 0.75],
        required=True,
        help="Probabilidad de las capas de Dropout"
    )

    args = parser.parse_args()

    return vars(args)  # Generamos un dict con los argumentos


def get_model_name(args):
    exp_name = "model_{}_epochs_{}_train_base_layers_{}_drop_prob_{}"
    exp_name = exp_name.format(
        args.get('model_type'),
        args.get('epochs'),
        args.get('train_base_layers'),
        args.get('drop_prob')
    )

    return exp_name


def save_to_disk(x_data, y_data, usage, output_dir='cifar10_images'):
    """
    This function will resize your data using the specified output_size and
    save them to output_dir.

    x_data : np.ndarray
        Array with images.

    y_data : np.ndarray
        Array with labels.

    usage : str
        One of ['train', 'val', 'test'].

    output_dir : str
        Path to save data.
    """
    assert usage in ['train', 'val', 'test']

    # Set paths
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in np.unique(y_data):
        label_path = os.path.join(output_dir, usage, str(label))
        if not os.path.exists(label_path):
            os.makedirs(label_path)

    for idx, img in enumerate(x_data):
        bgr_img = img[..., ::-1]  # RGB -> BGR
        # label = y_data[idx][0]
        label = y_data[idx]
        img_path = os.path.join(
            output_dir, usage, str(label), 'img_{}.jpg'.format(idx)
        )
        retval = cv2.imwrite(img_path, bgr_img)

        assert retval, 'Problem saving image at index: {}'.format(idx)


def select_model(model_type, base_model, train_base_layers,
                 drop_prob, num_classes):
    for layer in base_model.layers:
        layer.trainable = train_base_layers

    # Obtenemos el la salida del Feature Vector del "base model" cargado
    x = base_model.output

    if model_type == '1':
        x = Dense(1024, activation='relu')(x)  # Agregamos una capa Densa
        x = Dropout(drop_prob)(x)              # Agregamos capa de Dropout
        print("Model Type 1: 2 Layers")
    elif model_type == '2':
        x = Dense(128, activation='relu')(x)   # Agregamos una capa Densa
        x = Dropout(drop_prob)(x)              # Agregamos capa de Dropout
        x = Dense(256, activation='relu')(x)   # Agregamos una capa Densa
        x = Dropout(drop_prob)(x)              # Agregamos capa de Dropout
        x = Dense(512, activation='relu')(x)   # Agregamos una capa Densa
        x = Dropout(drop_prob)(x)              # Agregamos capa de Dropout
        x = Dense(1024, activation='relu')(x)  # Agregamos una capa Densa
        x = Dropout(drop_prob)(x)              # Agregamos capa de Dropout
        print("Model Type 2: 8 Layers")

    # Añadimos capa de clasificacion, usando 'Softmax'
    predictions = Dense(num_classes, activation='softmax')(x)

    # Creamos el modelo que sera entrenado
    model = Model(
        inputs=base_model.input,
        outputs=predictions
    )

    return model


if __name__ == '__main__':

    args = read_args()  # Dict of Arguments

    EPOCHS = args.get('epochs')
    MODEL_TYPE = args.get('model_type')
    TRAIN_BASE_LAYERS = bool(args.get('train_base_layers'))
    DROP_PROB = args.get('drop_prob')
    MODEL_NAME = get_model_name(args)

    EXPERIMENT_NAME = MODEL_NAME

    print(MODEL_NAME)
    print(args)

    PATH_GENERAL = '/users/mferreyra/DiploDatos2018/Computer_Vision/Lab/Mario'

    # 1. Load data and split in training / validation / testing sets

    CIFAR_IMG_ROWS = 32
    CIFAR_IMG_COLS = 32

    # Cifar-10 class names
    # We will create a dictionary for each type of label
    # This is a mapping from the int class name to their corresponding
    # string class name
    LABELS = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    # Load dataset from keras
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # y_train = y_train.reshape((50000,))
    # y_test = y_test.reshape((10000,))

    # # Split in Train, Val sets
    # TEST_SIZE = 0.2
    # RANDOM_STATE = 1995

    # #########################################
    # # [COMPLETE]                            #
    # # Split training set in train/val sets  #
    # # Use the sampling method that you want #
    # #########################################
    # X_train, X_val, y_train, y_val = train_test_split(
    #     x_train,
    #     y_train,
    #     test_size=TEST_SIZE,
    #     random_state=RANDOM_STATE
    # )

    # X_test, y_test = x_test, y_test

    # save_to_disk(
    #     X_train, y_train, 'train',
    #     output_dir=os.path.join(PATH_GENERAL, 'cifar10_images')
    # )
    # save_to_disk(
    #     X_val, y_val, 'val',
    #     output_dir=os.path.join(PATH_GENERAL, 'cifar10_images')
    # )
    # save_to_disk(
    #     X_test, y_test, 'test',
    #     output_dir=os.path.join(PATH_GENERAL, 'cifar10_images')
    # )

    # 2. Load CNN MobileNet
    # Some constants
    NET_IMG_ROWS = 128
    NET_IMG_COLS = 128
    CHANNELS = 3
    NUM_CLASSES = 10

    base_model = MobileNet(
        input_shape=(NET_IMG_ROWS, NET_IMG_COLS, CHANNELS),  # Input image size
        include_top=False,                                   # Drop classification layer
        weights='imagenet',                                  # Use imagenet pre-trained weights
        pooling='avg'                                        # Global AVG pooling for the output vector
    )

    # 3. Adapt CNN to our problem
    # Train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers

    model = select_model(
        MODEL_TYPE, base_model, TRAIN_BASE_LAYERS, DROP_PROB, NUM_CLASSES
    )
    for layer in base_model.layers:
        layer.trainable = TRAIN_BASE_LAYERS

    # Obtenemos el la salida del Feature Vector del "base model" cargado
    x = base_model.output

    x = Dense(1024, activation='relu')(x)  # Agregamos una capa Densa
    x = Dropout(DROP_PROB)(x)              # Agregamos capa de Dropout

    # Añadimos capa de clasificacion, usando 'Softmax'
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Creamos el modelo que sera entrenado
    model = Model(
        inputs=base_model.input,
        outputs=predictions
    )

    # 4. Data augmentation techniques
    # Training data generator
    datagen_train = ImageDataGenerator(
        rescale=1./255,         # We also can make a rescale on the data
        horizontal_flip=True,
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    # Validation data generator
    datagen_val = ImageDataGenerator(rescale=1./255)

    generator_train = datagen_train.flow_from_directory(
        os.path.join(PATH_GENERAL, 'cifar10_images/train'),
        target_size=(NET_IMG_ROWS, NET_IMG_COLS),
        class_mode='categorical',
        batch_size=32
    )

    generator_val = datagen_val.flow_from_directory(
        os.path.join(PATH_GENERAL, 'cifar10_images/val'),
        target_size=(NET_IMG_ROWS, NET_IMG_COLS),
        class_mode='categorical',
        batch_size=32
    )

    # 5. Add some Keras callbacks
    # EXP_ID = os.path.join(PATH_GENERAL, EXPERIMENT_NAME)
    EXP_ID = os.path.join(PATH_GENERAL, 'experiments')
    if not os.path.exists(EXP_ID):
        os.makedirs(EXP_ID)

    PATH_EXP_NAME = os.path.join(EXP_ID, EXPERIMENT_NAME)
    if not os.path.exists(PATH_EXP_NAME):
        os.makedirs(PATH_EXP_NAME)

    PATH_WEIGHT = os.path.join(PATH_EXP_NAME, 'weights')
    if not os.path.exists(PATH_WEIGHT):
        os.makedirs(PATH_WEIGHT)

    callbacks_list = [
        # ModelCheckpoin: Save the model after every epoch
        ModelCheckpoint(
            filepath=os.path.join(
                PATH_WEIGHT, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            ),
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='auto'
        ),
        # TensorBoard: Basic visualizations.
        TensorBoard(
            log_dir=os.path.join(
                PATH_EXP_NAME, 'logs'
            ),
            write_graph=True,
            write_images=False
        ),
        # CSVLogger: Callback that streams epoch results to a csv file.
        CSVLogger(
            filename=os.path.join(
                PATH_EXP_NAME, '{}.csv'.format(MODEL_NAME)
            ),
            separator=',',
            append=True
        ),
    ]

    # 6. Setup optimization algorithm with their hyperparameters
    # Compile the model (should be done after setting layers to non-trainable)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 7. Train Model!
    BATCH_SIZE = 32

    train_model = model.fit_generator(
        generator_train,
        epochs=EPOCHS,
        validation_data=generator_val,
        steps_per_epoch=generator_train.n // BATCH_SIZE,
        validation_steps=generator_val.n // BATCH_SIZE,
        callbacks=callbacks_list,
    )

    # 8. Save Model!
    model.save(
        os.path.join(PATH_EXP_NAME, '{}.h5'.format(MODEL_NAME))
    )
