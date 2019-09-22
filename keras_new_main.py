from pathlib import Path

import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from sklearn.metrics import roc_auc_score

import load_data as ld
from AdamW import AdamW
from load_data import FINDINGS
from keras.callbacks import LearningRateScheduler
import keras_generators as gen
import yaml
import argparse
import keras_utils
import keras_model
import os
import tensorflow as tf
from custom_accuracy import keras_accuracy, compute_image_probability_asloss, combine_predictions_each_batch, \
    compute_auc, compute_image_probability_production, \
    test_function_acc_class, accuracy_bbox_IOU, compute_image_probability_production_v2, compute_IoU, \
    acc_atelectasis, acc_cardiomegaly, acc_infiltration, acc_average, acc_effusion, \
    acc_mass, acc_nodule, acc_pneumonia, acc_pneumothorax, accuracy_asloss, accuracy_asproduction, keras_binary_accuracy
from custom_loss import keras_loss, test_compute_ground_truth_per_class_numpy
from keras_preds import predict_patch_and_save_results

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
results_path =config['results_path']
generated_images_path = config['generated_images_path']
processed_labels_path = config['processed_labels_path']
prediction_results_path = config['prediction_results_path']
train_mode = config['train_mode']
test_single_image = config['test_single_image']
trained_models_path = config['trained_models_path']


IMAGE_SIZE = 512
BATCH_SIZE = 1
BATCH_SIZE_TEST = 10
BOX_SIZE = 16

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
init_train_idx, df_train_init, df_val, df_bbox_test, df_class_test, df_bbox_train = ld.get_train_test(xray_df,
                                                                                                      random_state=1,
                                                                                                      do_stats=False,
                                                                                                      res_path = generated_images_path,
                                                                                                      label_col = 'Cardiomegaly_loc')
df_train=df_train_init
print('Training set: '+ str(df_train_init.shape))
print('Validation set: '+ str(df_val.shape))
print('Localization testing set: '+ str(df_bbox_test.shape))
print('Classification testing set: '+ str(df_class_test.shape))

# df_train = keras_utils.create_overlap_set_bootstrap(df_train_init, 0.9, seed=2)
init_train_idx = df_train['Dir Path'].index.values

# new_seed, new_tr_ind = ld.create_overlapping_test_set(init_train_idx, 1, 0.95,0.85, xray_df)
# print(new_tr_ind)

if train_mode:
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
    model.summary()

    model = keras_model.compile_model_accuracy(model)
    model = keras_model.compile_model_regularization(model)
    model = keras_model.compile_model_adamw(model, weight_dec=0.0001, batch_size=BATCH_SIZE,
                                            samples_epoch=train_generator.__len__()*BATCH_SIZE, epochs=60 )

    total_epochs = int(500000/train_generator.__len__())
    print("Total number of iterations: "+ str(total_epochs))

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=60,
                               mode='min',
                               verbose=1)

    checkpoint = ModelCheckpoint(
        filepath=trained_models_path+ 'best_model_single_patient_reg.h5',
        monitor='val_loss',
        verbose=2,
        save_best_only=True,
        mode='min',
        period=1
    )

    filepath = trained_models_path + "single_class_patient_reg-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint_on_epoch_end = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)
    print("df train STEPS")
    print(len(df_train)//BATCH_SIZE)
    print(train_generator.__len__())

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.__len__(),
        epochs=2,
        validation_data=valid_generator,
        validation_steps=valid_generator.__len__(),
        verbose=1,
        callbacks=[checkpoint, checkpoint_on_epoch_end, early_stop, lrate]
    )
    print("history")
    print(history.history)
    print(history.history['keras_accuracy'])
    np.save(results_path + 'train_info.npy', history.history)

    keras_utils.plot_train_validation(history.history['keras_accuracy'],
                                  history.history['val_keras_accuracy'],
                                  'train accuracy', 'validation accuracy', 'accuracy','accuracy', results_path)
    keras_utils.plot_train_validation(history.history['accuracy_asproduction'],
                                      history.history['val_accuracy_asproduction'],
                                      'train accuracy', 'validation accuracy',
                                      'accuracy_asproduction', 'accuracy_asproduction', results_path)

    keras_utils.plot_train_validation(history.history['loss'],
                                  history.history['val_loss'],
                                  'train loss', 'validation loss', 'loss', 'loss', results_path)
else:
    # model = load_model(trained_models_path+'best_model_single_100.h5', custom_objects={
    #     'keras_loss': keras_loss, 'keras_accuracy':keras_accuracy})
    # model = keras_model.compile_model(model)
    opt = AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.075,
          batch_size=BATCH_SIZE, samples_per_epoch=8000, epochs=46)

    model = load_model('C:/Users/s161590/Desktop/' + 'single_class_patient_adamw10-3-75-08-2.01.hdf5', custom_objects={
        'keras_loss': keras_loss, 'keras_accuracy': keras_accuracy, 'keras_binary_accuracy': keras_binary_accuracy,
        'accuracy_asloss': accuracy_asloss, 'accuracy_asproduction': accuracy_asproduction, 'AdamW':opt})

    # model = keras_model.compile_model_adamw(model, 0.075, 8000, 46)

    if not test_single_image:
        ########################################### TRAINING SET########################################################
        file_unique_name = 'train_set'
        test_set = df_train

        predict_patch_and_save_results(model, file_unique_name, df_train, skip_processing,
                                       BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)

        # ########################################### VALIDATION SET######################################################

        predict_patch_and_save_results(model, 'val_set', df_val, skip_processing,
                                       BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)


        ########################################### TESTING SET########################################################
        test_set = pd.concat([df_bbox_test, df_class_test])
        predict_patch_and_save_results(model, 'test_set', test_set, skip_processing,
                                       BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)

