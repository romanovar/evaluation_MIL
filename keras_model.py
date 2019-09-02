from keras.applications import ResNet50
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, ReLU
from keras.models import Model
from keras.optimizers import Adam
from custom_loss import keras_loss, keras_loss_reg
from custom_accuracy import keras_accuracy,acc_atelectasis, acc_cardiomegaly, acc_effusion, acc_infiltration, acc_mass, \
    acc_nodule, acc_pneumonia,  acc_pneumothorax,  acc_average, keras_binary_accuracy, accuracy_asloss, accuracy_asproduction
from keras import regularizers

def build_model(weight_reg):
    #base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    last = base_model.output

    downsamp = MaxPooling2D(pool_size=1, strides=1, padding='Valid')(last)

    recg_net = Conv2D(512, kernel_size=(3,3), padding='same',  kernel_regularizer= regularizers.l2(weight_reg))(downsamp)
    recg_net = BatchNormalization()(recg_net)
    recg_net = ReLU()(recg_net)
    recg_net = Conv2D(1, (1,1), padding='same', activation='sigmoid',  kernel_regularizer= regularizers.l2(weight_reg))(recg_net) #, activity_regularizer=l2(0.001)

    model = Model(base_model.input, recg_net)

    return model


def step_decay(epoch, lr):
    '''
    :param epoch: current epoch
    :param lr: current learning rate
    :return: decay every 10 epochs the learning rate with 0.1
    '''
    decay = 0.1
    lrate = lr
    if(epoch%10==0):
        lrate = lr * decay
    return lrate


def compile_model(model):
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss=keras_loss,
                  metrics=[keras_accuracy])
    return model


def compile_model_accuracy(model):
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss=keras_loss,
                  metrics=[keras_accuracy, keras_binary_accuracy, accuracy_asloss, accuracy_asproduction])
    return model

def compile_model_regularization(model):
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss=keras_loss_reg,
                  metrics=[keras_accuracy])

    return model