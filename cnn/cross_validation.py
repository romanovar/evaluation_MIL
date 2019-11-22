import os
from pathlib import Path

import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model

import cnn.nn_architecture.keras_generators as gen
from cnn.nn_architecture import keras_model
from cnn import keras_utils
import cnn.preprocessor.load_data as ld
from cnn.nn_architecture.custom_performance_metrics import keras_accuracy, accuracy_asloss, accuracy_asproduction, keras_binary_accuracy
from cnn.nn_architecture.custom_loss import keras_loss
from cnn.keras_preds import predict_patch_and_save_results

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
    generated_images_path = config['generated_images_path']
    processed_labels_path = config['processed_labels_path']
    prediction_results_path = config['prediction_results_path']
    train_mode = config['train_mode']
    test_single_image = config['test_single_image']
    trained_models_path = config['trained_models_path']

    if skip_processing:
        xray_df = ld.load_csv(processed_labels_path)
        print('Cardiomegaly label division')
        print(xray_df['Cardiomegaly'].value_counts())
    else:
        label_df = ld.get_classification_labels(classication_labels_path, False)
        processed_df = ld.preprocess_labels(label_df, image_path)
        xray_df = ld.couple_location_labels(localization_labels_path, processed_df, ld.PATCH_SIZE, results_path)
    print(xray_df.shape)
    print("Splitting data ...")


    class_name = 'Cardiomegaly'
    CV_SPLITS = 5
    for split in range(0, CV_SPLITS):

        df_train, df_val, df_test, _,_, _ = ld.get_train_test_CV(xray_df, CV_SPLITS, split, random_seed=1,  label_col=class_name)

        print('Training set: ' + str(df_train.shape))
        print('Validation set: ' + str(df_val.shape))
        print('Localization testing set: ' + str(df_test.shape))
        if train_mode:
            ############################################ TRAIN ###########################################################
            train_generator = gen.BatchGenerator(
                instances=df_train.values,
                batch_size=BATCH_SIZE,
                net_h=IMAGE_SIZE,
                net_w=IMAGE_SIZE,
                norm=keras_utils.normalize,
                box_size=BOX_SIZE,
                processed_y=skip_processing)

            valid_generator = gen.BatchGenerator(
                instances=df_val.values,
                batch_size=BATCH_SIZE,
                net_h=IMAGE_SIZE,
                net_w=IMAGE_SIZE,
                box_size=BOX_SIZE,
                norm=keras_utils.normalize,
                processed_y=skip_processing)

            model = keras_model.build_model()

            model = keras_model.compile_model_accuracy(model)
            # model = keras_model.compile_model_regularization(model)
            # model = keras_model.compile_model_adamw(model, weight_dec=0.0001, batch_size=BATCH_SIZE,
            #                                         samples_epoch=train_generator.__len__()*BATCH_SIZE, epochs=60 )

            #   checkpoint on every epoch is not really needed here, CALLBACK REMOVED from the generator
            # filepath = trained_models_path + "CV_patient_split_"+str(split)+"_-{epoch:02d}-{val_loss:.2f}.hdf5"
            # checkpoint_on_epoch_end = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

            # lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)
            print("df train STEPS")
            print(len(df_train)//BATCH_SIZE)
            print(train_generator.__len__())

            history = model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_generator.__len__(),
                epochs=40,
                validation_data=valid_generator,
                validation_steps=valid_generator.__len__(),
                verbose=1
                # callbacks=[lrate]
            )
            filepath = trained_models_path + class_name +"CV_"+str(split)+"_nov.hdf5"
            model.save(filepath)
            print("history")
            print(history.history)
            print(history.history['keras_accuracy'])
            np.save(results_path + 'train_info_'+str(split)+'.npy', history.history)

            keras_utils.plot_train_validation(history.history['keras_accuracy'],history.history['val_keras_accuracy'],
                                          'train accuracy', 'validation accuracy', 'CV_accuracy'+str(split), 'accuracy',
                                              results_path)
            keras_utils.plot_train_validation(history.history['accuracy_asproduction'],
                                              history.history['val_accuracy_asproduction'],
                                              'train accuracy', 'validation accuracy',
                                              'CV_accuracy_asproduction'+str(split), 'accuracy_asproduction', results_path)

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
                                           BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)
        else:
            files_found=0
            for file in Path(trained_models_path).glob("single_class_patient_reg-02-0.10" + "*.hdf5"):
                files_found += 1

            assert files_found == 1, "No model found/ Multiple models found, not clear which to use "
            model = load_model( str(file),
                                custom_objects={
                                    'keras_loss': keras_loss, 'keras_accuracy': keras_accuracy,
                                    'keras_binary_accuracy': keras_binary_accuracy,
                                    'accuracy_asloss': accuracy_asloss, 'accuracy_asproduction': accuracy_asproduction})
            model = keras_model.compile_model_accuracy(model)

            predict_patch_and_save_results(model, "test_set_CV"+(str(split)), df_test, skip_processing,
                                           BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)

