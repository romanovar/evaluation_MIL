import load_data as ld
from keras.applications import ResNet50V2
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback, Callback
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, ReLU
import os
from keras.models import Model
from keras.optimizers import Adam
import keras_generators as gen
import custom_loss as cl
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt


def build_model():
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

    #base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    last = base_model.output

    downsamp = MaxPooling2D(pool_size=1, strides=1, padding='Valid')(last)

    recg_net = Conv2D(512, kernel_size=(3,3), padding='same')(downsamp)
    recg_net = BatchNormalization()(recg_net)
    recg_net = ReLU()(recg_net)
    recg_net = Conv2D(14, (1,1), padding='same', activation='sigmoid')(recg_net) #, activity_regularizer=l2(0.001)

    model = Model(base_model.input, recg_net)

    return model


def step_decay(epoch, lr):
    '''
    :param epoch: current epoch
    :param lr: current learning rate
    :return: decay every 10 epochs the learning rate with 0.1
    '''
    decay = 0.1
    # lrate = lr*decay
    lrate = lr
    if(epoch%10==0):
        lrate = lr * decay
    return lrate


def compile_model(model):
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  #loss='binary_crossentropy',
                  loss=cl.keras_loss,  # Call the loss function with the selected layer
                  metrics=[cl.keras_accuracy])

    return model


