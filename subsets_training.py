import numpy as np
import load_data as ld
import pandas as pd
from keras.callbacks import LearningRateScheduler
import keras_generators as gen
import yaml
import argparse
import keras_utils
import keras_model
import os
from keras_preds import predict_patch_and_save_results

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    print(xray_df.shape)
    print('Cardiomegaly label division')
    print(xray_df['Cardiomegaly'].value_counts())
else:
    label_df = ld.get_classification_labels(classication_labels_path, False)
    processed_df = ld.preprocess_labels(label_df, image_path)
    xray_df = ld.couple_location_labels(localization_labels_path, processed_df, ld.PATCH_SIZE, results_path)
print(xray_df.shape)
print("Splitting data ...")

class_name = "Cardiomegaly"
overlap_ratio = 0.95

## USING CV SPLIT TO USE A SPECIFIC TRAIN AND TEST DATA

CV_SPLITS = 5
number_classifiers = 5
for split in range(0, CV_SPLITS):
    df_train, df_val, df_test, \
    df_bbox_train, df_bbox_test, train_only_class = ld.get_train_test_CV(xray_df, CV_SPLITS, split, random_seed=1,
                                                                         label_col=class_name, ratio_to_keep=None)

    print('Training set: ' + str(df_train.shape))
    print('Validation set: ' + str(df_val.shape))
    print('Localization testing set: ' + str(df_test.shape))
    seeds = np.random.randint(low=100, high=1000, size=number_classifiers)
    print(seeds)
    np.save(results_path + 'subsets_seed_CV' + str(split) + '_' + str(number_classifiers), seeds)
    train_ind_coll = []
    for curr_classifier in range(0, number_classifiers):
        class_train_subset = ld.get_train_subset(train_only_class, df_bbox_train.shape[0],
                                                 random_seed=seeds[curr_classifier], ratio_to_keep=overlap_ratio)
        print("new subset is :" + str(class_train_subset.shape))
        train_ind_coll.append(class_train_subset)
        df_train = pd.concat([df_bbox_train, class_train_subset])

        if train_mode:
            ##O##O##_##O#O##_################################ TRAIN ###########################################################
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
            lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)

            print("df train STEPS")
            print(len(df_train) // BATCH_SIZE)
            print(train_generator.__len__())

            history = model.fit_generator(
                generator=train_generator,
                steps_per_epoch=train_generator.__len__(),
                epochs=10,
                validation_data=valid_generator,
                validation_steps=valid_generator.__len__(),
                verbose=1,
                callbacks=[lrate]
            )
            filepath = trained_models_path +'subset_'+class_name+"_CV"+ str(split)+'_'+ str(curr_classifier)+'_'+\
                       str(overlap_ratio)+".hdf5"
            model.save(filepath)
            print("history")
            print(history.history)
            print(history.history['keras_accuracy'])
            np.save(results_path + 'train_info_'+ str(split)+'_'+ str(curr_classifier)+'_'+
                                           str(overlap_ratio)+ '.npy', history.history)

            keras_utils.plot_train_validation(history.history['keras_accuracy'], history.history['val_keras_accuracy'],
                                              'train accuracy', 'validation accuracy', 'CV_accuracy' + str(split),
                                              'accuracy',
                                              results_path)
            keras_utils.plot_train_validation(history.history['accuracy_asproduction'],
                                              history.history['val_accuracy_asproduction'],
                                              'train accuracy', 'validation accuracy',
                                              'CV_accuracy_asproduction' + str(split), 'accuracy_asproduction',
                                              results_path)

            keras_utils.plot_train_validation(history.history['loss'], history.history['val_loss'], 'train loss',
                                              'validation loss', 'CV_loss' + str(split), 'loss', results_path)

            ############################################    PREDICTIONS      #############################################
            ########################################### TRAINING SET########################################################
            # predict_patch_and_save_results(model, 'train_set_CV' + str(split), df_train, skip_processing,
            #                                BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)
            #
            ########################################### VALIDATION SET######################################################
            # predict_patch_and_save_results(model, 'val_set_CV' + str(split), df_val, skip_processing,
            #                                BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)

            ########################################### TESTING SET########################################################
            predict_patch_and_save_results(model, 'subset_test_set_CV' + str(split)+'_'+ str(curr_classifier)+'_'+
                                           str(overlap_ratio), df_test,
                                           skip_processing, BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)