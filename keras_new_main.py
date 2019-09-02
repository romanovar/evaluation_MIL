from pathlib import Path

import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from sklearn.metrics import roc_auc_score

import load_data as ld
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
BATCH_SIZE = 10
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
                                                                                                      random_state=0,
                                                                                                      do_stats=True,
                                                                                                      res_path = generated_images_path)
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
    # model.summary()

    model = keras_model.compile_model_accuracy(model)
    #model = keras_model.compile_model_regularization(model)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=15,
                               mode='min',
                               verbose=1)

    checkpoint = ModelCheckpoint(
        filepath=trained_models_path+ 'best_model_single_100.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )

    filepath = trained_models_path + "single_class-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint_on_epoch_end = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

    lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)
    print("df train STEPS")
    print(len(df_train)//BATCH_SIZE)
    print(train_generator.__len__())

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.__len__(),
        epochs=100,
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

    model = load_model(trained_models_path + 'single_class-25-2.96.hdf5', custom_objects={
        'keras_loss': keras_loss, 'keras_accuracy': keras_accuracy, 'keras_binary_accuracy': keras_binary_accuracy,
        'accuracy_asloss': accuracy_asloss, 'accuracy_asproduction': accuracy_asproduction})

    model = keras_model.compile_model_accuracy(model)

    if not test_single_image:
        ########################################### TRAINING SET########################################################
        file_unique_name = 'train_set'
        test_set = df_train

        predict_patch_and_save_results(model, file_unique_name, df_train, skip_processing,
                                       BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)

        # ########################################### VALIDATION SET######################################################

        predict_patch_and_save_results(model, 'val_set', df_val, skip_processing,
                                       BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)
        # file_unique_name = 'val_set'
        # val_set = df_val

        # val_generator = gen.BatchGenerator(
        #     instances=val_set.values,
        #     batch_size=BATCH_SIZE_TEST,
        #     net_h=IMAGE_SIZE,
        #     net_w=IMAGE_SIZE,
        #     box_size=BOX_SIZE,
        #     norm=keras_utils.normalize,
        #     processed_y=skip_processing)
        #
        # predictions = model.predict_generator(test_generator, steps=test_generator.__len__(), workers=1)
        # np.save(prediction_results_path + 'predictions_' + file_unique_name, predictions)
        #
        # all_img_ind = []
        # all_patch_labels = []
        # for batch_ind in range(test_generator.__len__()):
        #     x, y = test_generator.__getitem__(batch_ind)
        #     y_cast = y.astype(np.float32)
        #     res_img_ind = test_generator.get_batch_image_indices(batch_ind)
        #     all_img_ind = combine_predictions_each_batch(res_img_ind, all_img_ind, batch_ind)
        #     all_patch_labels = combine_predictions_each_batch(y_cast, all_patch_labels, batch_ind)
        #
        # np.save(prediction_results_path + 'image_indices_' + file_unique_name, all_img_ind)
        # np.save(prediction_results_path + 'patch_labels_' + file_unique_name, all_patch_labels)

        ########################################### TESTING SET########################################################
        test_set = pd.concat([df_bbox_test, df_class_test])
        predict_patch_and_save_results(model, 'test_set', test_set, skip_processing,
                                       BATCH_SIZE_TEST, BOX_SIZE, IMAGE_SIZE, prediction_results_path)


        # file_unique_name = 'test_set'
        # test_set = pd.concat([df_bbox_test, df_class_test])
        #
        # test_generator = gen.BatchGenerator(
        #     instances=test_set.values,
        #     batch_size=BATCH_SIZE_TEST,
        #     net_h=IMAGE_SIZE,
        #     net_w=IMAGE_SIZE,
        #     box_size=BOX_SIZE,
        #     norm=keras_utils.normalize,
        #     processed_y=skip_processing,
        #     shuffle=False)
        #
        # predictions = model.predict_generator(test_generator, steps=test_generator.__len__(), workers=1)
        # np.save(prediction_results_path + 'predictions_' + file_unique_name, predictions)
        #
        # all_img_ind = []
        # all_patch_labels = []
        # for batch_ind in range(test_generator.__len__()):
        #     x, y = test_generator.__getitem__(batch_ind)
        #     y_cast = y.astype(np.float32)
        #     res_img_ind = test_generator.get_batch_image_indices(batch_ind)
        #     all_img_ind = combine_predictions_each_batch(res_img_ind, all_img_ind, batch_ind)
        #     all_patch_labels = combine_predictions_each_batch(y_cast, all_patch_labels, batch_ind)
        #
        # np.save(prediction_results_path + 'image_indices_' + file_unique_name, all_img_ind)
        # np.save(prediction_results_path + 'patch_labels_' + file_unique_name, all_patch_labels)




    # if not test_single_image:
    #
    #     # test_set = pd.concat([df_bbox_test, df_class_test])
    #     test_set = pd.con
    # cat([df_train])
    #
    #     test_generator = gen.BatchGenerator(
    #         instances=test_set.values,
    #         batch_size=BATCH_SIZE_TEST,
    #         net_h=IMAGE_SIZE,
    #         net_w=IMAGE_SIZE,
    #         box_size=BOX_SIZE,
    #         norm=keras_utils.normalize,
    #         processed_y=skip_processing)
    #
    #     predictions = model.predict_generator(test_generator, steps= test_generator.__len__(), workers=1)
    #     predictions_tf = tf.cast(predictions, tf.float32)
    #     predictions_np = predictions.astype(float)
    #     np.save(results_path+'/predictions_XY', predictions)
    #
    #     print("PREDICTION SHAPE")
    #     print(len(predictions))
    #     res_df = create_empty_dataset_results(len(predictions))
    #
    #
    #
    #     patch_labels_all_batches = []
    #     img_ind_all_batches = []
    #     for batch_ind in range(test_generator.__len__()):
    #         print("new batch")
    #         print(batch_ind)
    #         x, y = test_generator.__getitem__(batch_ind)
    #         y = tf.cast(y, tf.float32)
    #         res_img_ind = test_generator.get_batch_image_indeces(batch_ind)
    #
    #         l_bound = batch_ind * BATCH_SIZE_TEST
    #         r_bound = (batch_ind + 1) * BATCH_SIZE_TEST
    #         img_labels_v1, img_prob_preds_v1 = compute_image_probability_asloss(predictions_tf[l_bound:r_bound, :, :, :], y,
    #                                                                             P=16)
    #         img_labels_v2, img_prob_preds_v2 = compute_image_probability_production(predictions_tf[l_bound:r_bound, :, :, :], y, P=16)
    #         img_labels_v3, img_prob_preds_v3 = compute_image_probability_production_v2(predictions_tf[l_bound:r_bound, :, :, :], y, P=16)
    #
    #         col_values = []
    #         ##### NEW
    #         for i in range(0, BATCH_SIZE_TEST):
    #
    #             col_values = [res_img_ind[i]]
    #             for ind in range(len(FINDINGS)):
    #                 find_pred = predictions[l_bound+i, :, :, ind]
    #                 col_values.append(find_pred)
    #                 res_df.loc[l_bound + i] = pd.Series(find_pred)
    #
    #                 sess = tf.Session()
    #
    #                 with sess.as_default():
    #                     fin_img_pred = img_prob_preds_v1[i, ind].eval()
    #                     res_df.loc[l_bound + i] =fin_img_pred
    #                     fin_img_label = img_labels_v1[i, ind].eval()
    #                     res_df.loc[l_bound + i] =fin_img_label
    #
    #                 print(fin_img_label)
    #                 print(type(fin_img_label))
    #
    #                 col_values.append(fin_img_pred)
    #                 col_values.append(fin_img_label)
    #                 print(type(fin_img_pred))
    #                 print("***************************")
    #                 print(col_values)
    #             # res_df.loc[l_bound+i] = col_values
    #             res_df.to_csv(results_path+ '/' + 'test.csv')
    #
    #     init_op = tf.global_variables_initializer()
    #
    #     image_labels_loss =0
    #     image_prob_predictions_loss = 0
    #     image_labels = 0
    #     image_prob_predictions = 0
    #     has_bbox_test = 0
    #     acc_pred_test=0
    #     acc_per_class_test = 0
    #
    #     # img_labels_v1, img_prob_preds_v1 = compute_image_probability_asloss(predictions, patch_labels_all_batches, P=16)
    #     # img_labels_v2, img_prob_preds_v2 = compute_image_probability_production(predictions, patch_labels_all_batches, P=16)
    #     # img_labels_v3, img_prob_preds_v3 = compute_image_probability_production_v2(predictions, patch_labels_all_batches, P=16)
    #
    #     with tf.Session() as sess:
    #         sess.run(init_op)
    #
    #         image_labels_loss, image_prob_predictions_loss = img_labels_v1.eval(), img_prob_preds_v1.eval()
    #         image_labels, image_prob_predictions  = img_labels_v2.eval(), img_prob_preds_v2.eval()
    #         image_labels_v3, image_prob_predictions_v3 = img_labels_v3.eval(), img_prob_preds_v3.eval()
    #
    #
    #     make_save_predictions(img_ind_all_batches, predictions, image_prob_predictions_loss, image_labels,
    #                               results_path, 'predictions_loss.csv')
    #     make_save_predictions(img_ind_all_batches, predictions, image_prob_predictions, image_labels,
    #                               results_path, 'predictions_production.csv')
    #     make_save_predictions(img_ind_all_batches, predictions, image_prob_predictions_v3, image_labels,
    #                               results_path, 'predictions_production_v3.csv')
    #
    #     ###################### EVALUATION METRICS ############################
    #     print("# EVALUATION METRICS ####")
    #     # print(predictions)
    #     # print(patch_labels_all_batches)
    #     # has_bbox, sum_bbox, acc_pred, acc_per_class = test_function_acc_class(predictions, patch_labels_all_batches, P=16, iou_threshold=0.1)
    #     # acc_per_class_V2 = accuracy_bbox_IOU(predictions, patch_labels_all_batches, P=16, iou_threshold=0.1)
    #     #
    #     #
    #     # print("accuracy type")
    #     # has_bbox_test, sum_bb, acc_pred_test, acc_per_class_test =  has_bbox.eval(), sum_bbox.eval(), acc_pred.eval(), acc_per_class.eval()
    #     # acc_per_class_V2_test = acc_per_class_V2.eval()
    #
    #
    #         # make_save_predictions(predictions, img_label_pred, class_label_ground_truth, test_set['Dir Path'], out_dir=results_path)
    #         # make_stack_predictions(predictions, img_label_pred, class_label_ground_truth, test_set['Dir Path'], out_dir=results_path)
    #     # print("Image labels ")
    #     # print(image_labels)
    #     # print("image labels loss")
    #     # print(image_labels_loss)
    #     #
    #     # # auc_all_classes = []
    #     # # for ind in range(7,len(ld.FINDINGS)):
    #     # #     auc_score = roc_auc_score(image_labels[:,ind], image_prob_predictions[:, ind])
    #     # #     auc_all_classes.append(auc_score)
    #
    #     auc_all_classes_loss = compute_auc(image_labels_loss, image_prob_predictions_loss)
    #     # print(auc_all_classes_loss)
    #     # print(type(auc_all_classes_loss[0]))
    #     auc_all_classes = compute_auc(image_labels, image_prob_predictions)
    #     # print(auc_all_classes)
    #     auc_all_classes_v3 = compute_auc(image_labels_v3, image_prob_predictions_v3)
    #     # print(auc_all_classes_v3)
    #
    #     keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_loss, 'auc_prob_as_loss.csv', results_path)
    #     keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes, 'auc_prob_active_patch.csv', results_path)
    #     #
    #     keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_v3, 'auc_prob_prod.csv', results_path)
    #
    #     # print("has bbox")
    #     # print(has_bbox_test)
    #     # print("sum")
    #     # print(sum_bb)
    #     # print("acc pred" )
    #     # print(acc_pred_test)
    #     # print("total acc")
    #     # print(acc_per_class_test)
    #     # print(acc_per_class_V2_test)
    #     test_predictions = model.evaluate_generator(
    #         test_generator,
    #         steps = test_generator.__len__()
    #     )
    #     #
    #     # print(test_bbox_predictions[0], test_bbox_predictions[1], test_bbox_predictions[2], test_bbox_predictions[3])
    #     #
    #     # print(test_classification_predictions[0], test_classification_predictions[1], test_classification_predictions[2],
    #     #       test_classification_predictions[3])
    #     print(test_predictions)
    #     print(model.metrics_names)
    #
    #
    #     keras_utils.save_evaluation_results(model.metrics_names, test_predictions, "eval_test.csv", results_path)
    #
    #
    #     print(test_predictions[0], test_predictions[1], test_predictions[2]) #, test_predictions[3])


