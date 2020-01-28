import argparse
import os

import numpy as np
import pandas as pd
import yaml
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.engine.saving import load_model

import cnn.nn_architecture.keras_generators as gen
import cnn.preprocessor.load_data_datasets as ldd
from cnn import keras_utils
from cnn.keras_preds import predict_patch_and_save_results, get_patch_labels_from_batches
from cnn.keras_utils import process_loaded_labels
from cnn.nn_architecture import keras_model
from cnn.nn_architecture.custom_loss import keras_loss_v2, keras_loss_v3
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, accuracy_asloss, accuracy_asproduction, \
    keras_binary_accuracy, combine_predictions_each_batch

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
BATCH_SIZE = 1
BATCH_SIZE_TEST = 1
BOX_SIZE = 16

if use_xray_dataset:
    df_train, df_val, df_bbox_test, df_class_test = ldd.load_process_xray14(config)
    df_test = pd.concat([df_bbox_test, df_class_test])
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
        shuffle=True,
        norm=keras_utils.normalize,
        box_size=BOX_SIZE,
        processed_y=skip_processing,
        interpolation=mura_interpolation)

    valid_generator = gen.BatchGenerator(
        instances=df_val.values,
        batch_size=BATCH_SIZE,
        shuffle=True,
        net_h=IMAGE_SIZE,
        net_w=IMAGE_SIZE,
        box_size=BOX_SIZE,
        norm=keras_utils.normalize,
        processed_y=skip_processing,
        interpolation=mura_interpolation)

    model = keras_model.build_model()
    model.summary()

    # model = keras_model.compile_model_adamw(model, weight_dec=0.0001, batch_size=BATCH_SIZE,
    #                                         samples_epoch=train_generator.__len__()*BATCH_SIZE, epochs=60 )
    # model = keras_model.compile_model_regularization(model)
    model = keras_model.compile_model_accuracy(model)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=60,
                               mode='min',
                               verbose=1)
    model_identifier = "_shoulder_001"
    checkpoint = ModelCheckpoint(
        filepath=trained_models_path + 'best_model' + model_identifier + "-{epoch:02d}-{val_loss:.2f}.hdf5",
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
        epochs=1,
        validation_data=valid_generator,
        validation_steps=valid_generator.__len__(),
        verbose=1,
        callbacks=[checkpoint]
    )
    print(model.get_weights()[2])
    print("history")
    print(history.history)
    print(history.history['keras_accuracy'])
    np.save(results_path + 'train_info' + model_identifier + '.npy', history.history)

    keras_utils.plot_train_validation(history.history['loss'],
                                      history.history['val_loss'],
                                      'train loss', 'validation loss', 'loss' + model_identifier,
                                      'loss' + model_identifier, results_path)
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
    ###################### old generator
    predict_patch_and_save_results(model, 'val_set', df_val, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                   mura_interpolation=mura_interpolation)
    predict_patch_and_save_results(model, 'train_set', df_train, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                   mura_interpolation=mura_interpolation)
    predict_patch_and_save_results(model, 'test_set', df_test, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                   mura_interpolation=mura_interpolation)
    # ############ GENERATOR
    # patch_labels = []
    # for i in range(0, df_val.iloc[:, 1].shape[0]):  # (15)
    #     g = process_loaded_labels(df_val.iloc[i, 1])
    #     patch_labels.append(g)
    # patch_labels_gt = np.asarray(patch_labels, dtype=np.float32).reshape(-1, 16, 16, 1)
    # np.save(prediction_results_path + 'patch_labels_val_set_TF', patch_labels_gt,
    #         allow_pickle=True)
    #
    # predictions = model.predict_generator(valid_generator, steps=valid_generator.__len__(), workers=1)
    # np.save(prediction_results_path + 'predictions_val_set_TF', predictions,
    #         allow_pickle=True)
    #
    # ######## Identity function
    # import numpy as np
    #
    # x = Input((16, 16,))
    # m = Model(x, x)
    # m.compile(loss=keras_loss_v3, optimizer='adam', metrics=[])
    #
    # pr = np.reshape(predictions, (-1, 16, 16,))
    # tr = np.reshape(patch_labels_gt, (-1, 16, 16,))
    #
    # score1 = m.evaluate(x=pr, y=tr, batch_size=1, verbose=0)
    # print("Identity")
    # print(score1)
else:
    # model = load_model(trained_models_path+'best_model_single_100.h5', custom_objects={
    #     'keras_loss': keras_loss, 'keras_accuracy':keras_accuracy})
    # model = keras_model.compile_model(model)
    # opt = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.075,
    #       batch_size=BATCH_SIZE, samples_per_epoch=8000, epochs=46)

    ######################################################################################
    # model = keras_model.compile_model_adamw(model, 0.075, 8000, 46)

    ######################################################################################
    # model = load_model(trained_models_path + '_shoulder_001_lrd-20-10.25.hdf5', custom_objects={
    #     'keras_loss_v2': keras_loss_v2, 'keras_accuracy': keras_accuracy,
    #     'keras_binary_accuracy': keras_binary_accuracy,
    #     'accuracy_asloss': accuracy_asloss, 'accuracy_asproduction': accuracy_asproduction})

    ######################################################################################
    model = load_model(trained_models_path + 'best_model_shoulder_001-08-0.74.hdf5', custom_objects={
        'keras_loss_v3': keras_loss_v3,  'keras_accuracy': keras_accuracy,
        'keras_binary_accuracy': keras_binary_accuracy,
        'accuracy_asloss': accuracy_asloss, 'accuracy_asproduction': accuracy_asproduction})

    ########################################### TRAINING SET########################################################

    #predict_patch_and_save_results(model, 'train_set', df_train, skip_processing,
    #                               BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path, mura_interpolation)

    # ########################################### VALIDATION SET######################################################

    predict_patch_and_save_results(model, 'val_set', df_val, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path, mura_interpolation)

    ########################################### TESTING SET########################################################
    predict_patch_and_save_results(model, 'test_set', df_test, skip_processing,
                                   BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path, mura_interpolation)
