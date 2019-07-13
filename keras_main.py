import time
start = time.time()

import keras as K
from keras.callbacks import Callback
# from keras.regularizers import l2

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print(K.eval(lr_with_decay))

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


ON_SERVER = False
SKIP_PROCESSING = False


IMAGE_SIZE = 512
BATCH_SIZE = 3
BOX_SIZE = 16


SERVER_PATH_C ="/home/rnromanova/scripts/Data_Entry_2017.csv"
SERVER_PATH_L ="/home/rnromanova/scripts/Bbox_List_2017.csv"
# SERVER_PATH_I = "/home/rnromanova/XRay14/images/"
SERVER_PATH_I = "/home/rnromanova/XRay14/test_images"
SERVER_OUT = "/home/rnromanova/scripts/out"

LOCAL_PATH_C = "C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv"
LOCAL_PATH_L = "C:/Users/s161590/Desktop/Data/X_Ray/Bbox_List_2017.csv"
LOCAL_PATH_I = "C:/Users/s161590/Desktop/Data/X_Ray/images-Copy/"
LOCAL_OUT = "C:/Users/s161590/Desktop/Data/X_Ray/out"

if ON_SERVER and not SKIP_PROCESSING:
    label_df = ld.get_classification_labels(SERVER_PATH_C, False)
    processed_df = ld.preprocess_labels(label_df, SERVER_PATH_I)
    xray_df = ld.couple_location_labels(SERVER_PATH_L , processed_df, ld.PATCH_SIZE, SERVER_OUT)
    out_dir= SERVER_OUT


elif not ON_SERVER and not SKIP_PROCESSING:
    label_df = ld.get_classification_labels(LOCAL_PATH_C, False)
    processed_df = ld.preprocess_labels(label_df, LOCAL_PATH_I)
    # _, processed_df = ld.load_process_png_v2(label_df, LOCAL_PATH_I)
    xray_df = ld.couple_location_labels(LOCAL_PATH_L , processed_df, ld.PATCH_SIZE, LOCAL_OUT)
    out_dir = LOCAL_OUT


elif ON_SERVER and SKIP_PROCESSING:
    test = ld.load_csv(SERVER_OUT+'/'+'processed_Y.csv')
    out_dir = SERVER_OUT

elif not ON_SERVER and SKIP_PROCESSING:
    xray_df = ld.load_csv(LOCAL_OUT+'/'+'processed_Y.csv')
    out_dir= LOCAL_OUT

# filtering the columns in the split of the train and test
print("Splitting data ...")
init_train_idx, df_train, df_val, df_bbox_test, df_class_test = ld.get_train_test(xray_df, random_state=0)
# seed, new_train_idx = ld.create_overlapping_test_set(init_train_idx, 0, 0.95, 0.9, xray_df)
print("Training set")
print(df_train.shape)
print(df_train["Dir Path"])

print("Validation set")
print(df_val.shape)
print(df_val["Dir Path"])

print("Test set")
print(df_class_test.shape)
print(df_class_test['Dir Path'])
print(df_bbox_test.shape)
print(df_bbox_test['Dir Path'])
end = time.time()
print("total execution time")
print(end - start)

#normalize between [-1, 1]
def normalize(im):
    return 2*(im/255) -1


# print(X.shape)
train_generator = gen.BatchGenerator(
            instances=df_bbox_test.values,
            batch_size=BATCH_SIZE,
            net_h=IMAGE_SIZE,
            net_w=IMAGE_SIZE,
            # net_crop=IMAGE_SIZE,
            norm=normalize,
            box_size = BOX_SIZE,
            processed_y= SKIP_PROCESSING
        )

valid_generator = gen.BatchGenerator(
    instances=df_val.values,
    batch_size=BATCH_SIZE,
    net_h=IMAGE_SIZE,
    net_w=IMAGE_SIZE,
    box_size = BOX_SIZE,
    # net_crop=IMAGE_SIZE,
    norm=normalize,
    processed_y=SKIP_PROCESSING

)



def build_model():
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

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


lrate = LearningRateScheduler(step_decay, verbose=1)

def compile_model(model):
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  #loss='binary_crossentropy',
                  loss=cl.keras_loss,  # Call the loss function with the selected layer
                  metrics=[cl.keras_accuracy])

    return model


model = build_model()
model.summary()

model = compile_model(model)



#################################################### TRAIN ##########################################
early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0001,
                                   patience=10,
                                   mode='min',
                                   verbose=1)

checkpoint = ModelCheckpoint(
    filepath='model_nodule_bbox.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    period=1
)

def on_epoch_end(self, epoch):
    if (epoch%10==0):
        print(K.eval(self.model.optimizer.lr))

history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(df_train)//BATCH_SIZE,
            epochs=3,
            validation_data=valid_generator,
            validation_steps=len(df_val)//BATCH_SIZE,
            verbose=2,
            callbacks = [checkpoint, early_stop, lrate]
            # callbacks=[checkpoint, early_stop],

        )

# list all data in history
print(history.history.keys())
plt.ioff()
# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['keras_accuracy'])
plt.plot(history.history['val_keras_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
if os.path.isfile(out_dir+ '/accuracy_training.png'):
   os.remove(out_dir+ '/accuracy_training.png')
plt.savefig(out_dir+ '/accuracy_training.png')
plt.clf()

# summarize history for loss
plt.ioff()
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
if os.path.isfile(out_dir+ '/loss_training.png'):
   os.remove(out_dir+ '/loss_training.png')
plt.savefig(out_dir+'/loss_training.png')

