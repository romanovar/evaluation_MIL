from keras.applications import ResNet50V2
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, ReLU
from keras.models import Model
from keras.optimizers import Adam
from custom_loss import keras_loss
from custom_accuracy import keras_accuracy, AUC_class1, AUC_class2, AUC_class3,keras_accuracy_asloss, keras_accuracy_revisited, keras_auc_score, \
    keras_auc_v3, keras_AUC



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


# this function allows additional evaluation metrices to be added
def compile_model_on_load(model):
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss=keras_loss,  # Call the loss function with the selected layer
                  metrics=[keras_accuracy])#, AUC_class1, AUC_class2, AUC_class3]) #, keras_accuracy_revisited, keras_accuracy_asloss])
    return model