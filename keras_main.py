import load_data as ld
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization, ReLU
import os
from keras.models import Model
from keras.optimizers import Adam
import keras_generators as gen
import custom_loss as cl
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ON_SERVER = False
IMAGE_SIZE = 512
BATCH_SIZE = 8
BOX_SIZE = 16


SERVER_PATH_C ="/home/rnromanova/scripts/Data_Entry_2017.csv"
SERVER_PATH_L ="/home/rnromanova/scripts/Bbox_List_2017.csv"
SERVER_PATH_I = "/home/rnromanova/XRay14/images/batch1"
SERVER_OUT = "/home/rnromanova/scripts/out"

LOCAL_PATH_C = "C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv"
LOCAL_PATH_L = "C:/Users/s161590/Desktop/Data/X_Ray/Bbox_List_2017.csv"
LOCAL_PATH_I = "C:/Users/s161590/Desktop/Data/X_Ray/images/"
LOCAL_OUT = "C:/Users/s161590/Desktop/Data/X_Ray/out"

if ON_SERVER:
    label_df = ld.get_classification_labels(SERVER_PATH_C, False)
    X, Y = ld.load_process_png_v2(label_df, SERVER_PATH_I)
    Y = ld.couple_location_labels(SERVER_PATH_L , Y, ld.PATCH_SIZE, SERVER_OUT)

else:
    label_df = ld.get_classification_labels(LOCAL_PATH_C, False)
    _, Y = ld.load_process_png_v2(label_df, LOCAL_PATH_I)
    Y = ld.couple_location_labels(LOCAL_PATH_L , Y, ld.PATCH_SIZE, LOCAL_OUT)

XY = ld.keep_index_and_diagnose_columns(Y)


# print(XY)
def normalize(im):
    im = im / 255
    return im


# print(X.shape)
train_generator = gen.BatchGenerator(
            instances=XY.values,
            batch_size=BATCH_SIZE,
            net_h=IMAGE_SIZE,
            net_w=IMAGE_SIZE,
            # net_crop=IMAGE_SIZE,
            norm=normalize,
            box_size = BOX_SIZE
        )

valid_generator = gen.BatchGenerator(
    instances=XY.values,
    batch_size=BATCH_SIZE,
    net_h=IMAGE_SIZE,
    net_w=IMAGE_SIZE,
    box_size = BOX_SIZE,
    # net_crop=IMAGE_SIZE,
    norm=normalize
)



def build_model():

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    last = base_model.output

    downsamp = MaxPooling2D(pool_size=1, strides=1, padding='Valid')(last)

    recg_net = Conv2D(512, kernel_size=(3,3), padding='same')(downsamp)
    recg_net = BatchNormalization()(recg_net)
    recg_net = ReLU()(recg_net)
    recg_net = Conv2D(14, (1,1), padding='same', activation='sigmoid')(recg_net)

    model = Model(base_model.input, recg_net)

    return model


def compile_model(model):
    optimizer = Adam(lr=0.0001)
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

model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(XY)//BATCH_SIZE,
            epochs=100,
            validation_data=valid_generator,
            validation_steps=len(XY)//BATCH_SIZE,
            verbose=1,
            callbacks=[checkpoint, early_stop],
            workers=4,
            max_queue_size=16
        )

