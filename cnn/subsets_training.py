import numpy as np
from cnn.preprocessor import load_data as ld
import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from cnn.nn_architecture import keras_generators as gen
from cnn import keras_utils
from cnn.nn_architecture import keras_model
import os
from cnn.keras_preds import predict_patch_and_save_results
from cnn.preprocessor.load_data import load_xray, split_xray_cv
from cnn.preprocessor.load_data_mura import load_mura, split_data_cv, filter_rows_on_class, get_train_subset_mura, \
    filter_rows_and_columns
from cnn.preprocessor.load_data_pascal import load_pascal, construct_train_test_cv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train_on_subsets(config):
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

    nr_epochs = config['nr_epochs']
    lr = config['lr']
    reg_weight = config['reg_weight']
    pooling_operator = config['pooling_operator']

    IMAGE_SIZE = 512
    BATCH_SIZE = 1
    BATCH_SIZE_TEST = 1
    BOX_SIZE = 16

    overlap_ratio = 0.95
    CV_SPLITS = 5
    number_classifiers = 5
    # this should have the same length as the number of classifiers
    subset_seeds = [1234, 5678, 9012, 3456, 7890]

    if use_xray_dataset:
        xray_df = load_xray(skip_processing, processed_labels_path, classication_labels_path, image_path,
                            localization_labels_path, results_path, class_name)
    elif use_pascal_dataset:
        pascal_df = load_pascal(pascal_image_path)
    else:
        df_train_val, test_df_all_classes = load_mura(skip_processing, mura_processed_train_labels_path,
                                                      mura_processed_test_labels_path, mura_train_img_path,
                                                      mura_train_labels_path, mura_test_labels_path, mura_test_img_path)


    for split in range(0, CV_SPLITS):
        if use_xray_dataset:
            df_train, df_val, df_test, df_bbox_train, \
            df_bbox_test, train_only_class = split_xray_cv(xray_df, CV_SPLITS,
                                                           split, class_name)
        elif use_pascal_dataset:
            df_train, df_val, df_test = construct_train_test_cv(pascal_df, CV_SPLITS, split)
        else:
            df_train, df_val = split_data_cv(df_train_val, CV_SPLITS, split, random_seed=1, diagnose_col=class_name,
                                             ratio_to_keep=None)
            df_test = filter_rows_and_columns(test_df_all_classes, class_name)

        for curr_classifier in range(0, number_classifiers):
            if split == 1:
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

            if train_mode and split == 1:
                ##O##O##_##O#O##_################################ TRAIN ###########################################################
                train_generator = gen.BatchGenerator(
                    instances=df_train_subset.values,
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
