import argparse
import os
import random as rn

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.engine.saving import load_model
from numpy.random import seed

import cnn.nn_architecture.keras_generators as gen
import cnn.preprocessor.load_data_datasets as ldd
from cnn import keras_utils
from cnn.keras_preds import predict_patch_and_save_results
from cnn.keras_utils import set_dataset_flag, build_path_results, make_directory
from cnn.nn_architecture import keras_model
from cnn.nn_architecture.custom_loss import keras_loss_v3_nor
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, accuracy_asloss
from cnn.preprocessor.process_input import fetch_preprocessed_images_csv

np.random.seed(1)
tf.random.set_seed(2)
rn.seed(1)
seed(1)
os.environ['PYTHONHASHSEED']='1'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.keras.backend.clear_session()


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

resized_images_before_training = config['resized_images_before_training']
skip_processing = config['skip_processing_labels']
image_path = config['image_path']
results_path = config['results_path']
train_mode = config['train_mode']
dataset_name = config['dataset_name']
mura_interpolation = config['mura_interpolation']
nr_epochs = config['nr_epochs']
lr = config['lr']
reg_weight = config['reg_weight']
pooling_operator = config['pooling_operator']
class_name = config['class_name']

IMAGE_SIZE = 512
BATCH_SIZE = 10
BATCH_SIZE_TEST = 1
BOX_SIZE = 16

use_xray, use_pascal = set_dataset_flag(dataset_name)
script_suffix = 'exploratory_exp'
trained_models_path = build_path_results(results_path, dataset_name, pooling_operator, script_suffix= script_suffix,
                                         result_suffix='trained_models')
prediction_results_path = build_path_results(results_path, dataset_name, pooling_operator, script_suffix= script_suffix,
                                         result_suffix='predictions')
make_directory(trained_models_path)
make_directory(prediction_results_path)

if use_xray:
    if resized_images_before_training:
        xray_df = fetch_preprocessed_images_csv(image_path, 'processed_imgs')
        #todo: delete after testing
        # xray_df = xray_df[-50:]
    else:
        xray_df = ldd.load_process_xray14(config)
    df_train, df_val, df_test = ldd.split_filter_data(config, xray_df)

elif use_pascal:
    df_train, df_val, df_test = ldd.load_preprocess_pascal(config)
else:
    df_train, df_val, df_test = ldd.load_preprocess_mura(config)


if train_mode:
    tf.keras.backend.clear_session()
    train_generator = gen.BatchGenerator(
    instances=df_train.values,
    resized_image = resized_images_before_training,
    batch_size=BATCH_SIZE,
    net_h=IMAGE_SIZE,
    net_w=IMAGE_SIZE,
    shuffle=True,
    norm=keras_utils.normalize,
    box_size=BOX_SIZE,
    processed_y=skip_processing,
    interpolation=mura_interpolation)

    valid_generator = gen.BatchGenerator(
        instances=df_val.values,
        resized_image = resized_images_before_training,
        batch_size=BATCH_SIZE,
        shuffle=True,
        net_h=IMAGE_SIZE,
        net_w=IMAGE_SIZE,
        box_size=BOX_SIZE,
        norm=keras_utils.normalize,
        processed_y=skip_processing,
        interpolation=mura_interpolation)

    model = keras_model.build_model(reg_weight)
    model.summary()

    model = keras_model.compile_model_accuracy(model, lr, pooling_operator)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=60,
                               mode='min',
                               verbose=1)

    # checkpoint - saving a model for minimal validation loss reached
    # check if it is active in the callbacks of the fit() method
    best_model_checkpoint = ModelCheckpoint(
        filepath=trained_models_path + 'best_model' + class_name +"-{epoch:02d}-{val_loss:.2f}.hdf5",
        monitor='val_loss',
        verbose=2,
        save_best_only=True,
        mode='min',
        period=1
    )
    # checkpoint - saving a model at the end of the epoch
    filepath = trained_models_path + class_name + "-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint_on_epoch_end = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)
    
    dynamic_lrate = LearningRateScheduler(keras_model.dynamic_lr)
    print("df train STEPS")
    print(len(df_train) // BATCH_SIZE)
    print(train_generator.__len__())

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.__len__(),
        epochs=nr_epochs,
        validation_data=valid_generator,
        validation_steps=valid_generator.__len__(),
        verbose=1,
        callbacks=[best_model_checkpoint, dynamic_lrate]
    )
    print(model.get_weights()[2])
    print("history")
    print(history.history)
    print(history.history['keras_accuracy'])
    np.save(trained_models_path + 'train_info' + class_name + '.npy', history.history)

    keras_utils.plot_train_validation(history.history['loss'],
                                      history.history['val_loss'],
                                      'train loss', 'validation loss', 'loss',
                                      'loss', trained_models_path)

    settings = np.array({'lr: ': lr, 'reg_weight: ': reg_weight, 'pooling_operator: ':pooling_operator})
    np.save(trained_models_path + 'train_settings.npy', settings)

    ##### EVALUATE function

    print("evaluate validation")
    evaluate = model.evaluate_generator(
        generator=valid_generator,
        steps=valid_generator.__len__(),
        verbose=1)

    evaluate_train = model.evaluate_generator(
        generator=train_generator,
        steps=train_generator.__len__(),
        verbose=1)
    test_generator = gen.BatchGenerator(
        instances=df_test.values,
        resized_image = resized_images_before_training,
        batch_size=BATCH_SIZE,
        net_h=IMAGE_SIZE,
        net_w=IMAGE_SIZE,
        shuffle=True,
        norm=keras_utils.normalize,
        box_size=BOX_SIZE,
        processed_y=skip_processing,
        interpolation=mura_interpolation)

    evaluate_test = model.evaluate_generator(
        generator=test_generator,
        steps=test_generator.__len__(),
        verbose=1)
    print("Evaluate Train")
    print(evaluate_train)
    print("Evaluate Valid")
    print(evaluate)
    print("Evaluate test")
    print(evaluate_test)

    predict_patch_and_save_results(model, 'val_set', df_val, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                   mura_interpolation=mura_interpolation,
                                   resized_images_before_training = resized_images_before_training)
    predict_patch_and_save_results(model, 'train_set', df_train, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                   mura_interpolation=mura_interpolation,
                                   resized_images_before_training = resized_images_before_training)
    predict_patch_and_save_results(model, 'test_set', df_test, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                   mura_interpolation=mura_interpolation,
                                   resized_images_before_training=resized_images_before_training)

else:
    ######################################################################################
    # deserealize a model and do predictions with it
    model = load_model(trained_models_path + 'Cardiomegaly-01-14.67.hdf5', compile=True, custom_objects={
        'keras_loss_v3_nor': keras_loss_v3_nor,  'keras_accuracy': keras_accuracy,
        'accuracy_asloss': accuracy_asloss})

    ########################################### TRAINING SET########################################################

    predict_patch_and_save_results(model, 'train_set', df_train, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path, mura_interpolation,
                                   resized_images_before_training)

    # ########################################### VALIDATION SET######################################################

    predict_patch_and_save_results(model, 'val_set', df_val, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path, mura_interpolation,
                                   resized_images_before_training)

    ########################################### TESTING SET########################################################
    predict_patch_and_save_results(model, 'test_set', df_test, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path, mura_interpolation,
                                   resized_images_before_training)
