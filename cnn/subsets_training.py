from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.engine.saving import load_model

from cnn import keras_utils
from cnn.keras_preds import predict_patch_and_save_results
from cnn.nn_architecture import keras_generators as gen
from cnn.nn_architecture import keras_model
from cnn.nn_architecture.custom_loss import keras_loss_v3_nor
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, keras_binary_accuracy, accuracy_asloss, \
    accuracy_asproduction
from cnn.preprocessor import load_data as ld
from cnn.preprocessor.load_data import load_xray, split_xray_cv
from cnn.preprocessor.load_data_mura import load_mura, split_data_cv, get_train_subset_mura, \
    filter_rows_and_columns
from cnn.preprocessor.load_data_pascal import load_pascal, construct_train_test_cv
import tensorflow as tf
from tensorflow.keras import backend as K


def train_on_subsets(config, number_splits, CV_split_to_use, number_classifiers, subset_seeds, overlap_ratio):
    """
    Trains several classifiers with similar training set, while preserving test and validation set the same.
    The aim is to compare the performance of these classifiers later in stability module.
    The script takes a specific cross validation split of training, validation and testing set, and then drops a
    portion of the samples from the training set. Validation and test set are not changed - they are as the original
    split. Then the script trains a classifier with each of the training subsets.
    :param config: yaml config file
    :param number_splits: number of cross validation  splits used in cross validation (CV) (run_cross_validation.py)
    :param CV_split_to_use: specific CV split for defining  train/test/validation set. Value is between [0, number_splits-1]
    :param number_classifiers: number of classifiers to train
    :param subset_seeds: seeds used to drop observations from original training set.
    :param overlap_ratio:  ration of observations which are preserved from the original training set, defined by the
    specific CV split.
    :return: Returns saved .npy file for the predictions, image_indices and patch labels for the train/test/valid set for
    each subset.
    """
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
    pascal_image_path = config['pascal_image_path']
    use_pascal_dataset = config['use_pascal_dataset']
    resized_images_before_training = config['resized_images_before_training']


    nr_epochs = config['nr_epochs']
    lr = config['lr']
    reg_weight = config['reg_weight']
    pooling_operator = config['pooling_operator']

    IMAGE_SIZE = 512
    BATCH_SIZE = 10
    BATCH_SIZE_TEST = 1
    BOX_SIZE = 16


    if use_xray_dataset:
        if resized_images_before_training:
            xray_df = ld.load_csv(image_path+'/processed_imgs.csv')
        else:
            xray_df = load_xray(skip_processing, processed_labels_path, classication_labels_path, image_path,
                            localization_labels_path, results_path)
        xray_df = ld.filter_observations(xray_df, class_name, 'No Finding')

    elif use_pascal_dataset:
        pascal_df = load_pascal(pascal_image_path)
    else:
        df_train_val, test_df_all_classes = load_mura(skip_processing, mura_processed_train_labels_path,
                                                      mura_processed_test_labels_path, mura_train_img_path,
                                                      mura_train_labels_path, mura_test_labels_path, mura_test_img_path)


    for split in range(0, number_splits):
        if use_xray_dataset:
            df_train, df_val, df_test, df_bbox_train, \
            df_bbox_test, train_only_class = split_xray_cv(xray_df, number_splits,
                                                           split, class_name)
        elif use_pascal_dataset:
            df_train, df_val, df_test = construct_train_test_cv(pascal_df, number_splits, split)
        else:
            df_train, df_val = split_data_cv(df_train_val, number_splits, split, random_seed=1, diagnose_col=class_name,
                                             ratio_to_keep=None)
            df_test = filter_rows_and_columns(test_df_all_classes, class_name)

        for curr_classifier in range(0, number_classifiers):
            if split == CV_split_to_use:
                print("#####################################################")
                print("SPLIT :" + str(split))
                print("classifier #: " + str(curr_classifier))
                if use_xray_dataset:
                    class_train_subset = ld.get_train_subset_xray(train_only_class, df_bbox_train.shape[0],
                                                                  random_seed=subset_seeds[curr_classifier],
                                                                  ratio_to_keep=overlap_ratio)
                    print("new subset is :" + str(class_train_subset.shape))
                    df_train_subset = pd.concat([df_bbox_train, class_train_subset])
                    print(df_bbox_train.shape)
                    print(class_train_subset.shape)
                elif use_pascal_dataset:
                    df_train_subset = get_train_subset_mura(df_train, random_seed=subset_seeds[curr_classifier],
                                                            ratio_to_keep=overlap_ratio)
                else:
                    df_train_subset = get_train_subset_mura(df_train, random_seed=subset_seeds[curr_classifier],
                                                            ratio_to_keep=overlap_ratio)

            if train_mode and split == CV_split_to_use:
                tf.keras.backend.clear_session()
                K.clear_session()

                ##O##O##_##O#O##_################################ TRAIN ###########################################################
                train_generator = gen.BatchGenerator(
                    instances=df_train_subset.values,
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
                model = keras_model.compile_model_accuracy(model, lr, pooling_operator)
                lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)

                filepath = trained_models_path + "CV_patient_split_" + str(split) + "_-{epoch:02d}-{val_loss:.2f}.hdf5"
                checkpoint_on_epoch_end = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                                          mode='min')

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
                    callbacks=[checkpoint_on_epoch_end, lrate]
                )
                filepath = trained_models_path + 'subset_' + class_name + "_CV" + str(split) + '_' + str(
                    curr_classifier) + '_' + \
                           str(overlap_ratio) + ".hdf5"
                model.save(filepath)
                print("history")
                print(history.history)
                print(history.history['keras_accuracy'])
                np.save(results_path + 'train_info_' + str(split) + '_' + str(curr_classifier) + '_' +
                        str(overlap_ratio) + '.npy', history.history)

                settings = np.array({'lr: ': lr, 'reg_weight: ': reg_weight, 'pooling_operator: ': pooling_operator})
                np.save(results_path + 'train_settings.npy', settings)

                keras_utils.plot_train_validation(history.history['loss'], history.history['val_loss'], 'train loss',
                                                  'validation loss', 'CV_loss' + str(split)+ str(curr_classifier), 'loss', results_path)

                ############################################    PREDICTIONS      #############################################
                ########################################### TRAINING SET########################################################
                predict_patch_and_save_results(model, class_name+'_train_set_CV' + str(split)+'_'+ str(curr_classifier),
                                               df_train, skip_processing,
                                               BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                               mura_interpolation)

                ########################################## VALIDATION SET######################################################
                predict_patch_and_save_results(model, class_name+'_val_set_CV' + str(split)+'_'+ str(curr_classifier),
                                               df_val, skip_processing,
                                               BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                               mura_interpolation)

                ########################################### TESTING SET########################################################
                predict_patch_and_save_results(model, class_name+
                                               '_test_set_CV' + str(split) + '_' + str(curr_classifier) +
                                               str(class_name)+'_' +
                                               str(overlap_ratio), df_test,
                                               skip_processing, BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE,
                                               prediction_results_path, mura_interpolation)
            elif not train_mode:
                files_found = 0
                print(trained_models_path)
                for file_path in Path(trained_models_path).glob(
                        "CV_patient_split_" + str(curr_classifier) + "_-04" + "*.hdf5"):
                    print(file_path)
                    files_found += 1

                assert files_found == 1, "No model found/ Multiple models found, not clear which to use "
                print(str(files_found))
                ### OLD MODEL loading
                # model = load_model(str(file_path),
                #                    custom_objects={
                #                        'keras_loss_v2': keras_loss_v2, 'keras_accuracy': keras_accuracy,
                #                        'keras_binary_accuracy': keras_binary_accuracy,
                #                        'accuracy_asloss': accuracy_asloss,
                #                        'accuracy_asproduction': accuracy_asproduction})

                model = load_model(str(file_path),
                                   custom_objects={
                                       'keras_loss_v3_nor': keras_loss_v3_nor, 'keras_accuracy': keras_accuracy,
                                       'keras_binary_accuracy': keras_binary_accuracy,
                                       'accuracy_asloss': accuracy_asloss,
                                       'accuracy_asproduction': accuracy_asproduction})

                model = keras_model.compile_model_accuracy(model, lr, pooling_operator)

                predict_patch_and_save_results(model, "train_set_CV" + (str(split)), df_train, skip_processing,
                                               BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                               mura_interpolation, resized_images_before_training)
                predict_patch_and_save_results(model, "val_set_CV" + (str(split)), df_val, skip_processing,
                                               BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                               mura_interpolation, resized_images_before_training)
                predict_patch_and_save_results(model, "test_set_CV" + (str(split)), df_test, skip_processing,
                                               BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path,
                                               mura_interpolation,resized_images_before_training)
