from keras.applications import ResNet50
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, ReLU, Dropout
from keras.models import Model
from keras.optimizers import Adam

from cnn.nn_architecture.AdamW import AdamW
# from keras.optimizers_v2 import Adam
from cnn.nn_architecture.custom_loss import keras_loss, keras_loss_reg, keras_loss_v2
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, keras_binary_accuracy, accuracy_asloss, \
    accuracy_asproduction


def build_model():
    #base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    # base_model.trainable = False
    for layer in base_model.layers:
        layer.trainable = False
    count = 0
    for layer in base_model.layers:
        if 'res5' in layer.name: # or 'res4' in layer.name:
            layer.trainable = True
            count +=1
            print('trainable layer')
            print(count)
            print(layer.name)

    last = base_model.output

    downsamp = MaxPooling2D(pool_size=1, strides=1, padding='Valid')(last)

    recg_net = Conv2D(512, kernel_size=(3,3), padding='same')(downsamp)
    recg_net = BatchNormalization()(recg_net)
    recg_net = ReLU()(recg_net)
    recg_net = Conv2D(1, (1,1), padding='same', activation='sigmoid')(recg_net) #, activity_regularizer=l2(0.001)
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
    if(epoch%10==0) and (epoch > 0):
        lrate = lr * decay
    return lrate


def compile_model_adamw(model, weight_dec, batch_size, samples_epoch, epochs):
    optimizer = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,decay=0., weight_decay=weight_dec,
                      batch_size=batch_size, samples_per_epoch=samples_epoch, epochs=epochs)
    model.compile(optimizer=optimizer,
                  loss=keras_loss,
                  metrics=[keras_accuracy, keras_binary_accuracy, accuracy_asloss, accuracy_asproduction])
    return model


def compile_model(model):
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss=keras_loss,
                  metrics=[keras_accuracy])
    return model


def compile_model_accuracy(model):
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss=keras_loss_v2,
                  metrics=[keras_accuracy, keras_binary_accuracy, accuracy_asloss, accuracy_asproduction])
    return model


def compile_model_regularization(model):
    ## LR=0.01 but after the callback it is 0.001
    ## So LR is effectibely 0.001
    optimizer = Adam(lr=0.01)
    model.compile(optimizer=optimizer,
                  loss=keras_loss_reg,
                  metrics=[keras_accuracy, keras_binary_accuracy, accuracy_asloss, accuracy_asproduction])

    return model