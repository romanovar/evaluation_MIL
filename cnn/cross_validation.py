import os
from pathlib import Path

import numpy as np
import pandas as pd
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model

import cnn.nn_architecture.keras_generators as gen
from cnn.nn_architecture import keras_model
from cnn import keras_utils
import cnn.preprocessor.load_data as ld
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, accuracy_asloss, accuracy_asproduction, keras_binary_accuracy
from cnn.nn_architecture.custom_loss import keras_loss, keras_loss_v3, keras_loss_v3_nor
from cnn.keras_preds import predict_patch_and_save_results
from cnn.preprocessor.load_data_datasets import load_process_xray14
from cnn.preprocessor.load_data_mura import load_mura, split_data_cv, filter_rows_on_class, filter_rows_and_columns
from cnn.preprocessor.load_data_pascal import load_pascal, construct_train_test_cv
from cnn.preprocessor.process_input import fetch_preprocessed_images_csv
import tensorflow as tf
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

IMAGE_SIZE = 512
BATCH_SIZE = 10
BATCH_SIZE_TEST = 1
BOX_SIZE = 16


def cross_validation(config):

    skip_processing = config['skip_processing_labels']
    image_path = config['image_path']
    classication_labels_path = config['classication_labels_path']
    localization_labels_path = config['localization_labels_path']
    results_path = config['results_path']
    processed_labels_path = config['processed_labels_path']
    prediction_results_path = config['prediction_results_path']
    train_mode = config['train_mode']
    trained_models_path = config['trained_models_path']
    use_xray_dataset = config['use_xray_dataset']
    class_name = config['class_name']
    mura_test_img_path = config['mura_test_img_path']
    mura_train_labels_path = config['mura_train_labels_path']
    mura_train_img_path = config['mura_train_img_path']
    mura_test_labels_path= config['mura_test_labels_path']
    mura_processed_train_labels_path = config['mura_processed_train_labels_path']
    mura_processed_test_labels_path = config['mura_processed_test_labels_path']
    mura_interpolation = config['mura_interpolation']
    use_pascal_dataset = config['use_pascal_dataset']
    pascal_image_path = config['pascal_image_path']
    resized_images_before_training=config['resized_images_before_training']

    nr_epochs = config['nr_epochs']
    lr = config['lr']
    reg_weight = config['reg_weight']
    pooling_operator = config['pooling_operator']

    if use_xray_dataset:
        if resized_images_before_training:
            xray_df = fetch_preprocessed_images_csv(image_path, 'processed_imgs')
        else:
            xray_df = load_process_xray14(config)
    elif use_pascal_dataset:
        pascal_df = load_pascal(pascal_image_path)

    else:
        df_train_val, test_df_all_classes = load_mura(skip_processing, mura_processed_train_labels_path,
                                                      mura_processed_test_labels_path, mura_train_img_path,
                                                      mura_train_labels_path, mura_test_labels_path, mura_test_img_path)

    CV_SPLITS = 5
    for split in range(0, CV_SPLITS):

        if use_xray_dataset:
            df_train, df_val, df_test, _, _,_ = ld.split_xray_cv(xray_df, CV_SPLITS,
                                                                 split, class_name)

        elif use_pascal_dataset:
            df_train, df_val, df_test = construct_train_test_cv(pascal_df, CV_SPLITS, split)

        else:
            df_train, df_val = split_data_cv(df_train_val, CV_SPLITS, split, random_seed=1, diagnose_col=class_name,
                                             ratio_to_keep=None)
            # df_test = filter_rows_on_class(test_df_all_classes, class_name=class_name)
            df_test = filter_rows_and_columns(test_df_all_classes, class_name)

        if train_mode:
            tf.keras.backend.clear_session()
            K.clear_session()

            ############################################ TRAIN ###########################################################
            train_generator = gen.BatchGenerator(
                instances=df_train.values,
                resized_image=resized_images_before_training,
                batch_size=BATCH_SIZE,
                net_h=IMAGE_SIZE,
                net_w=IMAGE_SIZE,
                norm=keras_utils.normalize,
                box_size=BOX_SIZE,
                processed_y=skip_processing,
                interpolation=mura_interpolation,
                shuffle=True)

            valid_generator = gen.BatchGenerator(
                instances=df_val.values,
                resized_image=resized_images_before_training,
                batch_size=BATCH_SIZE,
                net_h=IMAGE_SIZE,
                net_w=IMAGE_SIZE,
                box_size=BOX_SIZE,
                norm=keras_utils.normalize,
                processed_y=skip_processing,
                interpolation=mura_interpolation,
                shuffle=True)

            model = keras_model.build_model(reg_weight)

            model = keras_model.compile_model_accuracy(model, lr, pool_op=pooling_operator)

            #   checkpoint on every epoch is not really needed here, CALLBACK REMOVED from the generator
            filepath = trained_models_path + "CV_patient_split_"+str(split)+"_-{epoch:02d}-{val_loss:.2f}.hdf5"
            checkpoint_on_epoch_end = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

            lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)
            print("df train STEPS")
            print(len(df_train)//BATCH_SIZE)
            print(train_generator.__len__())

            history = model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_generator.__len__(),
                epochs=nr_epochs,
                validation_data=valid_generator,
                validation_steps=valid_generator.__len__(),
                verbose=1
            )
            # filepath = trained_models_path + class_name +"CV_"+str(split)+"_nov.hdf5"
            # model.save(filepath)
            print("history")
            print(history.history)
            print(history.history['keras_accuracy'])
            np.save(results_path + 'train_info_'+str(split)+'.npy', history.history)

            settings = np.array({'lr: ': lr, 'reg_weight: ': reg_weight, 'pooling_operator: ': pooling_operator})
            np.save(results_path + 'train_settings.npy', settings)
            keras_utils.plot_train_validation(history.history['loss'], history.history['val_loss'], 'train loss',
                                              'validation loss', 'CV_loss'+str(split), 'loss', results_path)

            ############################################    PREDICTIONS      #############################################
            ########################################### TRAINING SET########################################################
            # predict_patch_and_save_results(model, 'train_set_CV'+str(split), df_train, skip_processing,
            #                                BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)
            #
            ########################################### VALIDATION SET######################################################
            # predict_patch_and_save_results(model, 'val_set_CV'+str(split), df_val, skip_processing,
            #                                BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)

            ########################################### TESTING SET########################################################
            predict_patch_and_save_results(model, 'test_set_'+ class_name+'_CV'+str(split), df_test, skip_processing,
                                           BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                           mura_interpolation, resized_images_before_training)
            predict_patch_and_save_results(model, 'train_set_' + class_name + '_CV' + str(split), df_train,
                                           skip_processing,
                                           BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                           mura_interpolation, resized_images_before_training)
            predict_patch_and_save_results(model, 'val_set_' + class_name + '_CV' + str(split), df_val,
                                           skip_processing,
                                           BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                           mura_interpolation, resized_images_before_training)
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
                resized_image=resized_images_before_training,
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
        else:
            files_found = 0
            print(trained_models_path)
            for file_path in Path(trained_models_path).glob("CV_patient_split_0"+str(split) + "*.hdf5"):
                print(file_path)
                files_found += 1

            assert files_found == 1, "No model found/ Multiple models found, not clear which to use "
            print(str(files_found))
            model = load_model(str(file_path),
                               custom_objects={
                                   'keras_loss_v3_nor': keras_loss_v3_nor, 'keras_accuracy': keras_accuracy,
                                   'keras_binary_accuracy': keras_binary_accuracy,
                                   'accuracy_asloss': accuracy_asloss, 'accuracy_asproduction': accuracy_asproduction})
            model = keras_model.compile_model_accuracy(model, lr, pooling_operator)

            predict_patch_and_save_results(model, "train_set_CV" + (str(split)), df_train, skip_processing,
                                           BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                           mura_interpolation, resized_images_before_training)
            predict_patch_and_save_results(model, "val_set_CV" + (str(split)), df_val, skip_processing,
                                           BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                           mura_interpolation, resized_images_before_training)
            predict_patch_and_save_results(model, "test_set_CV" + (str(split)), df_test, skip_processing,
                                           BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                           mura_interpolation, resized_images_before_training)


