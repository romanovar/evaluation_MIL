from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import MaxPooling2D, Conv2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from cnn.nn_architecture.custom_loss import keras_loss_v3_nor, keras_loss_v3_mean, \
    keras_loss_v3_lse, keras_loss_v3_lse01, keras_loss_v3_max
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, accuracy_asloss


def build_model(reg_weight):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    ## freezing layers
    #for layer in base_model.layers:
    #    layer.trainable = False
    # Unfeeezing only last ones
    # count = 0
    # for layer in base_model.layers:
    #     if 'res5' in layer.name:
    #         layer.trainable = True
    #         count +=1
    #         print('trainable layer')
    #         print(count)
    #         print(layer.name)

    last = base_model.output

    downsamp = MaxPooling2D(pool_size=1, strides=1, padding='Valid')(last)

    recg_net = Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', activity_regularizer=regularizers.l2(reg_weight))(downsamp)
    recg_net = BatchNormalization()(recg_net)
    recg_net = Conv2D(1, (1,1), padding='same', activation='sigmoid')(recg_net)
    model = Model(base_model.input, recg_net)
    
    return model


def step_decay(epoch, lr, decay=None):
    '''
    :param epoch: current epoch
    :param lr: current learning rate
    :return: decay every 10 epochs the learning rate with 0.1
    '''
    if not decay:
        return lr
    else:
        lrate = lr
        if(epoch%10==0) and (epoch > 0):
            lrate = lr * decay
        return lrate


def dynamic_lr(epoch):
    return 1e-5 * 10 **(epoch/20)


def compile_model_accuracy(model, lr, pool_op):
    loss_f ={'nor': keras_loss_v3_nor,
            'mean': keras_loss_v3_mean,
             'lse': keras_loss_v3_lse,
             'lse01': keras_loss_v3_lse01,
             'max': keras_loss_v3_max
     }
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer,
                  loss=loss_f[pool_op],
                  metrics=[keras_accuracy, accuracy_asloss])
    return model

