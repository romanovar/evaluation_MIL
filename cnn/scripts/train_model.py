import argparse
import os

import cnn.nn_architecture.keras_generators as gen
from cnn.nn_architecture import keras_model
import numpy as np
import pandas as pd
import yaml
from cnn.nn_architecture.custom_loss import keras_loss, keras_loss_v2
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, accuracy_asloss, accuracy_asproduction, \
    keras_binary_accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.engine.saving import load_model

from cnn import keras_utils
import cnn.preprocessor.load_data_datasets as ldd
from cnn.keras_preds import predict_patch_and_save_results

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
results_path = config['results_path']
prediction_results_path = config['prediction_results_path']
train_mode = config['train_mode']
trained_models_path = config['trained_models_path']
use_xray_dataset = config['use_xray_dataset']
mura_interpolation = config['mura_interpolation']
use_pascal_dataset = config['use_pascal_dataset']



IMAGE_SIZE = 512
BATCH_SIZE = 10
BATCH_SIZE_TEST = 1
BOX_SIZE = 16

if use_xray_dataset:
    df_train, df_val, df_bbox_test, df_class_test = ldd.load_process_xray14(config)
    test_set = pd.concat([df_bbox_test, df_class_test])
elif use_pascal_dataset:
    df_train, df_val, df_test = ldd.load_preprocess_pascal(config)
else:
    df_train, df_val, df_test = ldd.load_preprocess_mura(config)


if train_mode:
    train_generator = gen.BatchGenerator(
        instances=df_train.values,
        batch_size=BATCH_SIZE,
        net_h=IMAGE_SIZE,
        net_w=IMAGE_SIZE,
        norm=keras_utils.normalize,
        box_size=BOX_SIZE,
        processed_y=skip_processing,
        interpolation=mura_interpolation)

    valid_generator = gen.BatchGenerator(
        instances=df_val.values,
        batch_size=BATCH_SIZE,
        net_h=IMAGE_SIZE,
        net_w=IMAGE_SIZE,
        box_size=BOX_SIZE,
        norm=keras_utils.normalize,
        processed_y=skip_processing,
        interpolation=mura_interpolation)

    model = keras_model.build_model()
    model.summary()

    print(model.get_weights()[2])
    # model = keras_model.compile_model_adamw(model, weight_dec=0.0001, batch_size=BATCH_SIZE,
    #                                         samples_epoch=train_generator.__len__()*BATCH_SIZE, epochs=60 )
    #model = keras_model.compile_model_regularization(model)
    model = keras_model.compile_model_accuracy(model)

    total_epochs = int(500000 / train_generator.__len__())
    print("Total number of iterations: " + str(total_epochs))

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=60,
                               mode='min',
                               verbose=1)
    model_identifier = "_shoulder_001"
    checkpoint = ModelCheckpoint(
        filepath=trained_models_path + 'best_model' + model_identifier + '.h5',
        monitor='val_loss',
        verbose=2,
        save_best_only=True,
        mode='min',
        period=1
    )

    filepath = trained_models_path + model_identifier + "-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint_on_epoch_end = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)
    print("df train STEPS")
    print(len(df_train) // BATCH_SIZE)
    print(train_generator.__len__())

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.__len__(),
        epochs=50,
        validation_data=valid_generator,
        validation_steps=valid_generator.__len__(),
        verbose=1,
        callbacks=[checkpoint, checkpoint_on_epoch_end]
        # callbacks = [checkpoint, checkpoint_on_epoch_end, early_stop, lrate]

    )
    print(model.get_weights()[2])
    print("history")
    print(history.history)
    print(history.history['keras_accuracy'])
    np.save(results_path + 'train_info' + model_identifier + '.npy', history.history)

    keras_utils.plot_train_validation(history.history['keras_accuracy'],
                                      history.history['val_keras_accuracy'],
                                      'train accuracy', 'validation accuracy', 'accuracy' + model_identifier,
                                      'accuracy' + model_identifier, results_path)
    keras_utils.plot_train_validation(history.history['accuracy_asproduction'],
                                      history.history['val_accuracy_asproduction'],
                                      'train accuracy', 'validation accuracy',
                                      'accuracy_asproduction' + model_identifier,
                                      'accuracy_asproduction' + model_identifier, results_path)

    keras_utils.plot_train_validation(history.history['loss'],
                                      history.history['val_loss'],
                                      'train loss', 'validation loss', 'loss' + model_identifier,
                                      'loss' + model_identifier, results_path)
else:
    # model = load_model(trained_models_path+'best_model_single_100.h5', custom_objects={
    #     'keras_loss': keras_loss, 'keras_accuracy':keras_accuracy})
    # model = keras_model.compile_model(model)
    # opt = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.075,
    #       batch_size=BATCH_SIZE, samples_per_epoch=8000, epochs=46)
    # model = keras_model.compile_model_adamw(model, 0.075, 8000, 46)
    model = load_model(trained_models_path + '_shoulder_001_lrd-20-10.25.hdf5', custom_objects={
        'keras_loss_v2': keras_loss_v2, 'keras_accuracy': keras_accuracy, 'keras_binary_accuracy': keras_binary_accuracy,
        'accuracy_asloss': accuracy_asloss, 'accuracy_asproduction': accuracy_asproduction})



    ########################################### TRAINING SET########################################################
    file_unique_name = 'train_set'
    test_set = df_train

    predict_patch_and_save_results(model, file_unique_name, df_train, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)

    # ########################################### VALIDATION SET######################################################

    predict_patch_and_save_results(model, 'val_set', df_val, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)

    ########################################### TESTING SET########################################################
    predict_patch_and_save_results(model, 'test_set', test_set, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)
