# Setup one GPU for tensorflow (don't be greedy).
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, "0", "1", etc.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# https://keras.io/applications/#documentation-for-individual-models
import keras
from keras.applications.mobilenet import MobileNet
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.regularizers import l2
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import cv2
import argparse
import numpy as np
import tensorflow as tf
from utils import *
import collections

# Limit tensorflow gpu usage.
# Maybe you should comment this lines if you run tensorflow on CPU.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)


parser = argparse.ArgumentParser(description='Transfer learning with MobileNet')
parser.add_argument('--train_base_layers',
                    default='no',
                    type=str,
                    help='Finetune or not MobileNet pretrained layers.')
parser.add_argument('--pooling',
                    default='avg',
                    type=str,
                    help='Use of avg or max pooling strategy')
parser.add_argument('--drop',
                    default=0,
                    type=float,
                    help='Randomly fraction rate of input units to 0'
                         'at each update during training time')
parser.add_argument('--l2',
                    default=0.1,
                    type=float,
                    help='L2 kernel regularizer')
parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='Number of instances in each batch.')
parser.add_argument('--epochs',
                    default=10,
                    type=int,
                    help='Number of training epochs.')

args = parser.parse_args()


NET_IMG_ROWS = 128
NET_IMG_COLS = 128
CHANNELS = 3
NUM_CLASSES = 10
POOLING = args.pooling
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
TRAIN_BASE_LAYERS = args.train_base_layers == 'yes'  # True or False

# Training data generator
datagen_train = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2
)
# Validation data generator
datagen_val = ImageDataGenerator(
    rescale=1. / 255
)
generator_train = datagen_train.flow_from_directory('cifar10_images/train',
                                                    target_size=(NET_IMG_ROWS, NET_IMG_COLS),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')
# https://stackoverflow.com/questions/51305682/how-to-use-to-categorical-when-using-imagedatagenerator
generator_val = datagen_val.flow_from_directory('cifar10_images/val',
                                                target_size=(NET_IMG_ROWS, NET_IMG_COLS),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')
# Note that we have to resize our images to finetune the MobileNet CNN, this is done using
# the target_size argument in flow_from_directory. Remember to set the target_size to one of
# the valid listed here: [(128, 128), (160, 160), (192, 192), or (224, 224)].
############

print('Building model ...')

base_model = MobileNet(
    input_shape=(NET_IMG_ROWS, NET_IMG_COLS, CHANNELS),   # Input image size
    weights='imagenet',                                   # Use imagenet pre-trained weights
    include_top=False,                                    # Drop classification layer
    pooling=POOLING
)
############


# Having the CNN loaded, now we have to add some layers to adapt this network to our
# classification problem.
# We can choose to finetune just the new added layers, some particular layers or all the layer of the
# model. Play with different settings and compare the results.

for layer in base_model.layers:
    layer.trainable = TRAIN_BASE_LAYERS  # True or False

x = base_model.output
x = Dense(1024, activation='relu', kernel_regularizer=l2(args.l2))(x)
x = Dropout(args.drop)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
############

# Load and set some Keras callbacks here!
args = list(vars(args).items())
experiment_name = 'model'
for key, value in args:
    experiment_name += ('_' + str(key) + '_' + str(value))
if not os.path.exists('experiments'):
    os.makedirs('experiments')


# class ExpName(Callback): # No funciona: https://stackoverflow.com/questions/48488549/keras-append-to-logs-from-callback
#     def __init__(self, exp_name):
#         self.exp_name = exp_name

#     def on_epoch_end(self, epoch, logs):
#         logs['exp_name'] = self.exp_name

callbacks = [
    ModelCheckpoint(filepath=os.path.join('experiments', experiment_name,
                                          'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=False,
                    save_weights_only=False,
                    mode='auto'),
    TensorBoard(log_dir=os.path.join('experiments', experiment_name, 'logs'),
                write_graph=True,
                write_images=False),
    # ExpName(experiment_name),
    CSVLogger(filename=os.path.join('experiments', experiment_name, 'train_logs.csv'),
              separator=',',
              append=True)
]
###########

print('Compiling model ...')

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adadelta',
    metrics=['accuracy'])
############


# Use fit_generator to train your model.
print('Training model ...')
model.fit_generator(
    generator_train,
    epochs=EPOCHS,
    validation_data=generator_val,
    steps_per_epoch=generator_train.n // BATCH_SIZE,
    validation_steps=generator_val.n // BATCH_SIZE,
    callbacks=callbacks)
############
