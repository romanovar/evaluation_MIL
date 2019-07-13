import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
import load_data as ld
from keras.callbacks import LearningRateScheduler
import keras_generators as gen
import yaml
import argparse
import keras_utils
import keras_model
import os


start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

skip_processing = config['skip_processing_labels']
image_path = config['image_path']
classication_labels_path = config['classication_labels_path']
localization_labels_path = config['localization_labels_path']
results_path = config['results_path']
processed_labels_path = config['processed_labels_path']

IMAGE_SIZE = 512
BATCH_SIZE = 3
BOX_SIZE = 16

if skip_processing:
    print("im here")
    xray_df = ld.load_csv(processed_labels_path)
else:
    classication_labels_path = 'C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv'
    print("im false")
    print(classication_labels_path)
    print(type(classication_labels_path))
    label_df = ld.get_classification_labels(classication_labels_path, False)
    processed_df = ld.preprocess_labels(label_df, image_path)
    xray_df = ld.couple_location_labels(localization_labels_path, processed_df, ld.PATCH_SIZE, results_path)

print("Splitting data ...")
init_train_idx, df_train, df_val, df_bbox_test, df_class_test = ld.get_train_test(xray_df, random_state=0)

train_generator = gen.BatchGenerator(
    instances=df_bbox_test.values,
    batch_size=BATCH_SIZE,
    net_h=IMAGE_SIZE,
    net_w=IMAGE_SIZE,
    # net_crop=IMAGE_SIZE,
    norm=keras_utils.normalize,
    box_size=BOX_SIZE,
    processed_y=skip_processing)

valid_generator = gen.BatchGenerator(
    instances=df_val.values,
    batch_size=BATCH_SIZE,
    net_h=IMAGE_SIZE,
    net_w=IMAGE_SIZE,
    box_size=BOX_SIZE,
    # net_crop=IMAGE_SIZE,
    norm=keras_utils.normalize,
    processed_y=skip_processing)

model = keras_model.build_model()
model.summary()

model = keras_model.compile_model(model)

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.0001,
                           patience=10,
                           mode='min',
                           verbose=1)

checkpoint = ModelCheckpoint(
    filepath=results_path+ '/model_nodule_bbox.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    period=1
)

lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(df_train) // BATCH_SIZE,
    epochs=3,
    validation_data=valid_generator,
    validation_steps=len(df_val) // BATCH_SIZE,
    verbose=2,
    callbacks=[checkpoint, early_stop, lrate]
)

keras_utils.plot_train_validation(history.history['keras_accuracy'],
                                  history.history['val_keras_accuracy'],
                                  'model accuracy', 'accuracy', results_path)

keras_utils.plot_train_validation(history.history['loss'],
                                  history.history['val_loss'],
                                  'model loss', 'loss', results_path)
