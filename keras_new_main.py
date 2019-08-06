from pathlib import Path

import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from sklearn.metrics import roc_auc_score

import load_data as ld
from keras.callbacks import LearningRateScheduler
import keras_generators as gen
import yaml
import argparse
import keras_utils
import keras_model
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.image import pil_to_array
from custom_accuracy import keras_accuracy, compute_image_probability_asloss, combine_predictions_each_batch, \
    make_save_predictions, compute_auc, compute_image_probability_production, \
    test_function_acc_class, accuracy_bbox_IOU, compute_image_probability_production_v2, compute_IoU, \
    create_empty_dataset_results
from custom_loss import keras_loss, test_compute_ground_truth_per_class_numpy

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
results_path = config['results_path']
processed_labels_path = config['processed_labels_path']
train_mode = config['train_mode']
test_single_image = config['test_single_image']

IMAGE_SIZE = 512
BATCH_SIZE = 100
BATCH_SIZE = 2
BATCH_SIZE_TEST = 2
BOX_SIZE = 16

if skip_processing:
    xray_df = ld.load_csv(processed_labels_path)
else:
    label_df = ld.get_classification_labels(classication_labels_path, False)
    processed_df = ld.preprocess_labels(label_df, image_path)
    xray_df = ld.couple_location_labels(localization_labels_path, processed_df, ld.PATCH_SIZE, results_path)

print("Splitting data ...")
init_train_idx, df_train, df_val, df_bbox_test, df_class_test = ld.get_train_test(xray_df, random_state=0, do_stats=True, res_path =results_path)
# df_train, df_val, df_bbox_test, df_class_test = ld.get_train_test_strata(xray_df, random_state=0, do_stats=True, res_path=results_path)
print(init_train_idx)
init_train_idx = df_train['Dir Path'].index.values
# new_seed, new_tr_ind = ld.create_overlapping_test_set(init_train_idx, 1, 0.95,0.85, xray_df)
# print(new_tr_ind)

if train_mode:
    train_generator = gen.BatchGenerator(
        instances=df_bbox_test.values,
        batch_size=BATCH_SIZE,
        net_h=IMAGE_SIZE,
        net_w=IMAGE_SIZE,
        # net_crop=IMAGE_SIZE,
        norm=keras_utils.normalize,
        box_size=BOX_SIZE,
        processed_y=skip_processing)

    valid_generator = gen.BatchGenerator(
        instances=df_val.values,
        batch_size=BATCH_SIZE,
        net_h=IMAGE_SIZE,
        net_w=IMAGE_SIZE,
        box_size=BOX_SIZE,
        # net_crop=IMAGE_SIZE,
        norm=keras_utils.normalize,
        processed_y=skip_processing)

    model = keras_model.build_model()
    model.summary()

    model = keras_model.compile_model(model)

    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=10,
                               mode='min',
                               verbose=1)

    checkpoint = ModelCheckpoint(
        filepath=results_path+ '/model_nodule_bbox.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )

    lrate = LearningRateScheduler(keras_model.step_decay, verbose=1)
    print(len(df_train))
    print("df train")
    print(len(df_val))
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(df_train) // BATCH_SIZE,
        epochs=2,
        validation_data=valid_generator,
        validation_steps=len(df_val) // BATCH_SIZE,
        verbose=1,
        callbacks=[checkpoint, early_stop, lrate]
    )
    print("history")
    print(history.history)
    print(history.history['keras_accuracy'])
    keras_utils.plot_train_validation(history.history['keras_accuracy'],
                                  history.history['val_keras_accuracy'],
                                  'model accuracy', 'training_accuracy', 'accuracy','accuracy', results_path)

    keras_utils.plot_train_validation(history.history['loss'],
                                  history.history['val_loss'],
                                  'model loss', 'training_loss', 'loss', 'loss', results_path)
else:

    model = load_model(results_path+'/trained_model_v1.h5', custom_objects={'keras_loss': keras_loss, 'keras_accuracy':keras_accuracy})

    model = keras_model.compile_model_on_load(model)

    # if skip_processing:
    # process labels in the same way as in the batch generators
    if not test_single_image:

        # test_set = pd.concat([df_bbox_test, df_class_test])
        test_set = pd.concat([df_train])

        test_generator = gen.BatchGenerator(
            instances=test_set.values,
            batch_size=BATCH_SIZE_TEST,
            net_h=IMAGE_SIZE,
            net_w=IMAGE_SIZE,
            box_size=BOX_SIZE,
            norm=keras_utils.normalize,
            processed_y=skip_processing)

        predictions = model.predict_generator(test_generator, steps= test_generator.__len__(), workers=1)
        print("PREDICTION SHAPE")
        print(len(predictions))
        res_df = create_empty_dataset_results()
        res_df.to_csv(results_path+ '/' + 'test.csv')

        # for i in range(len(predictions)):
            # res_df.loc[i] = ['']
            # res_df['Dir Path'] = res_img_ind

        patch_labels_all_batches = []
        img_ind_all_batches = []
        for batch_ind in range(test_generator.__len__()):
            print("new batch")
            print(batch_ind)
            x, y = test_generator.__getitem__(batch_ind)
            res_img_ind = test_generator.get_batch_image_indeces(batch_ind)
            for i in range(BATCH_SIZE_TEST):
                res_df['Dir Path'] = res_img_ind
                res_df['']
                res_df[ld.FINDINGS[0] + '_pred'] = pd.Series([predictions[n, :, :, 0] for n in range(predictions_df.shape[0])])

            # res_df['']
            img_ind_all_batches = combine_predictions_each_batch(res_img_ind, img_ind_all_batches, batch_ind)
            patch_labels_all_batches = combine_predictions_each_batch(y, patch_labels_all_batches, batch_ind)
        patch_labels_all_batches =tf.cast(patch_labels_all_batches, tf.float32)

        init_op = tf.global_variables_initializer()

        image_labels_loss =0
        image_prob_predictions_loss = 0
        image_labels = 0
        image_prob_predictions = 0
        has_bbox_test = 0
        acc_pred_test=0
        acc_per_class_test = 0

        img_labels_v1, img_prob_preds_v1 = compute_image_probability_asloss(predictions, patch_labels_all_batches, P=16)
        img_labels_v2, img_prob_preds_v2 = compute_image_probability_production(predictions, patch_labels_all_batches, P=16)
        img_labels_v3, img_prob_preds_v3 = compute_image_probability_production_v2(predictions, patch_labels_all_batches, P=16)

        with tf.Session() as sess:
            sess.run(init_op)

            image_labels_loss, image_prob_predictions_loss = img_labels_v1.eval(), img_prob_preds_v1.eval()
            image_labels, image_prob_predictions  = img_labels_v2.eval(), img_prob_preds_v2.eval()
            image_labels_v3, image_prob_predictions_v3 = img_labels_v3.eval(), img_prob_preds_v3.eval()


        make_save_predictions(img_ind_all_batches, predictions, image_prob_predictions_loss, image_labels,
                                  results_path, 'predictions_loss.csv')
        make_save_predictions(img_ind_all_batches, predictions, image_prob_predictions, image_labels,
                                  results_path, 'predictions_production.csv')
        make_save_predictions(img_ind_all_batches, predictions, image_prob_predictions_v3, image_labels,
                                  results_path, 'predictions_production_v3.csv')

        ###################### EVALUATION METRICS ############################
        print("# EVALUATION METRICS ####")
        print(predictions)
        print(patch_labels_all_batches)
        has_bbox, sum_bbox, acc_pred, acc_per_class = test_function_acc_class(predictions, patch_labels_all_batches, P=16, iou_threshold=0.1)
        acc_per_class_V2 = accuracy_bbox_IOU(predictions, patch_labels_all_batches, P=16, iou_threshold=0.1)


        print("accuracy type")
        has_bbox_test, sum_bb, acc_pred_test, acc_per_class_test =  has_bbox.eval(), sum_bbox.eval(), acc_pred.eval(), acc_per_class.eval()
        acc_per_class_V2_test = acc_per_class_V2.eval()


            # make_save_predictions(predictions, img_label_pred, class_label_ground_truth, test_set['Dir Path'], out_dir=results_path)
            # make_stack_predictions(predictions, img_label_pred, class_label_ground_truth, test_set['Dir Path'], out_dir=results_path)
        # print("Image labels ")
        # print(image_labels)
        # print("image labels loss")
        # print(image_labels_loss)
        #
        # # auc_all_classes = []
        # # for ind in range(7,len(ld.FINDINGS)):
        # #     auc_score = roc_auc_score(image_labels[:,ind], image_prob_predictions[:, ind])
        # #     auc_all_classes.append(auc_score)

        auc_all_classes_loss = compute_auc(image_labels_loss, image_prob_predictions_loss)
        # print(auc_all_classes_loss)
        # print(type(auc_all_classes_loss[0]))
        auc_all_classes = compute_auc(image_labels, image_prob_predictions)
        # print(auc_all_classes)
        auc_all_classes_v3 = compute_auc(image_labels_v3, image_prob_predictions_v3)
        # print(auc_all_classes_v3)

        keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_loss, 'auc_prob_as_loss.csv', results_path)
        keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes, 'auc_prob_active_patch.csv', results_path)
        #
        keras_utils.save_evaluation_results(ld.FINDINGS, auc_all_classes_v3, 'auc_prob_prod.csv', results_path)

        # print("has bbox")
        # print(has_bbox_test)
        # print("sum")
        # print(sum_bb)
        # print("acc pred" )
        # print(acc_pred_test)
        # print("total acc")
        # print(acc_per_class_test)
        # print(acc_per_class_V2_test)
        test_predictions = model.evaluate_generator(
            test_generator,
            steps = test_generator.__len__()
        )
        #
        # print(test_bbox_predictions[0], test_bbox_predictions[1], test_bbox_predictions[2], test_bbox_predictions[3])
        #
        # print(test_classification_predictions[0], test_classification_predictions[1], test_classification_predictions[2],
        #       test_classification_predictions[3])
        print(test_predictions)
        print(model.metrics_names)


        keras_utils.save_evaluation_results(model.metrics_names, test_predictions, "eval_test.csv", results_path)


        print(test_predictions[0], test_predictions[1], test_predictions[2]) #, test_predictions[3])

    else:

        dir_img = "img_5/"
        img_ind = "00000211_010"
        string_dir_path = dir_img+img_ind+".png"
        path_2 = Path(image_path+dir_img+img_ind+".png")
        single_image_df = df_bbox_test[df_bbox_test['Dir Path'] == str(path_2)]

        img_ind2 = "00000181_061"
        string_dir_path2 = dir_img + img_ind2 + ".png"
        path_img2 = Path(image_path + dir_img + img_ind2 + ".png")
        second_image_df = df_bbox_test[df_bbox_test['Dir Path'] == str(path_img2)]


        dir_img2 = 'img_3/'
        img_ind3 = "00000147_001"
        string_dir_path2 = dir_img2 + img_ind3 + ".png"
        path_img3 = Path(image_path + dir_img2 + img_ind3 + ".png")
        third_image_df = df_bbox_test[df_bbox_test['Dir Path'] == str(path_img3)]


        test_df = pd.concat([single_image_df, second_image_df, third_image_df])

        test_generator = gen.BatchGenerator(
            instances=third_image_df.values,
            batch_size=1,
            net_h=IMAGE_SIZE,
            net_w=IMAGE_SIZE,
            box_size=BOX_SIZE,
            norm=keras_utils.normalize,
            processed_y=skip_processing)

        prediction = model.predict_generator(test_generator, steps=test_generator.__len__())

        #TODO: to remove only FOR test purpose
        _, y_labels = test_generator.__getitem__(0)
        y_labels = y_labels.astype(np.float32)

        class_label_ground_truth, img_label_pred = compute_image_probability_asloss(prediction[0], y_labels[0], P=16)
        # cl2, img_label2 = compute_image_probability_asloss(prediction[1], y_labels[1], P=16)
        cl_batch, img_label_batch = compute_image_probability_asloss(prediction, y_labels, P=16)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        iou = compute_IoU(prediction, y_labels, 16)
        acc_per_class = accuracy_bbox_IOU(prediction, y_labels, 16, 0.1)

        # init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            class_label_ground_truth =class_label_ground_truth.eval()
            img_label_pred = img_label_pred.eval()
            # cl2 = cl2.eval()
            # img_label2 = img_label2.eval()

            cl_batch = cl_batch.eval()
            img_label_batch = img_label_batch.eval()
            # auc = auc_score_tf(cl_batch[:, 0], img_label_batch[:, 0] )
            # auc_score = auc.eval()
            iou_score = iou.eval()
            acc_per_class_sc = acc_per_class.eval()

        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        auc_sc = roc_auc_score(y_true, y_scores)
        print(auc_sc)


        print("image 0")
        print("GT classses: ")
        print(class_label_ground_truth)
        print("prediction: ")
        print(img_label_pred)

        # print("imgae 2: ")
        # print(cl2, img_label2)

        print("batch")
        print(cl_batch, img_label_batch)
        print("class")
        print(img_label_batch[:, 0])
        print(cl_batch[:, 0])

        # print("auc")
        #
        #
        #
        # for i in [0, 4, 8, 11, 13]:
        #     auc_sc_T = roc_auc_score(cl_batch[:, i], img_label_batch[:, i] )
        #     print(img_label_batch[:, i])
        #     print(cl_batch[:, i])
        #     print("sklearn auc")
        #     print("class: "+str(i))
        #     print(auc_sc_T)
        #     print(type(cl_batch[:, i]))
        #
        # result_np = np.concatenate((cl_batch[:, 0], cl_batch[:, 4], cl_batch[:, 8]))
        # labels_np = np.concatenate((img_label_batch[:, 0], img_label_batch[:, 4], img_label_batch[:, 8]))
        # test_conc = np.concatenate((labels_np, img_label_batch[:, 0]))
        #
        # print("this is TEEST")
        # print(test_conc.shape)
        # conc = np.stack((result_np, labels_np),axis=1)
        # print(conc)
        #
        # #### AUC on not sorted arrays
        # auc_score_not_sorted = roc_auc_score(conc[:, 0], conc[:, 1])
        # print("nt sorted auc")
        # print(auc_score_not_sorted)
        #
        #
        # #### sorting ndarray
        # res_conc_sorting = conc[conc[:, 0].argsort()]
        # print(res_conc_sorting)
        # print("Sorted auc ndarray")
        # auc_score_sorted = roc_auc_score(res_conc_sorting[:, 0], res_conc_sorting[:, 1])
        #
        # print(auc_score_sorted)
        #
        # ### TF AUC score
        # auc_tf = tf.metrics.auc(res_conc_sorting[:, 0], res_conc_sorting[:, 1])
        # print(auc_tf.value)
        #
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # # init_op = tf.initialize_all_variables()
        # with tf.Session() as sess:
        #     sess.run(init_op)
        #     auc_tf = auc_tf.value
        # print(auc_tf)
        # ######
        test_stats = model.evaluate_generator(test_generator, steps=1)
        print(test_stats)
############################################################################
        # x, y = test_generator.__getitem__(0)
        img_label, img_prob = compute_image_probability_production(prediction[:, :, :, :], y_labels.astype(np.float32), P=16)


        img_label_np = 0
        img_prob_np = 0

        # initialize the variable
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            img_label_np =img_label.eval()
            img_prob_np = img_prob.eval()

        # keras_utils.visualize_single_image_all_classes(single_image_df, '00000211_010', results_path,
        #                                                prediction, img_prob_np, img_label_np, acc_per_class_sc,
        #                                                iou_score)
        # keras_utils.visualize_single_image_all_classes(second_image_df, '00000181_061', results_path,
        #                                                prediction, img_prob_np, img_label_np, acc_per_class_sc,
        #                                                iou_score)

        keras_utils.visualize_single_image_all_classes(third_image_df, '00000147_001', results_path,
                                                       prediction, img_prob_np, img_label_np, acc_per_class_sc,
                                                       iou_score)

        # for row in second_image_df.values:
        #     labels_df = []
        #
        #     for i in range(1, row.shape[0]):  # (15)
        #         g = ld.process_loaded_labels_tf(row[i])
        #
        #         sum_active_patches, class_label_ground, has_bbox = test_compute_ground_truth_per_class_numpy(g, 16 * 16)
        #         print("sum active patches: " + str(sum_active_patches))
        #         print("class label: " + str(class_label_ground))
        #         print("Has bbox:" + str(has_bbox))
        #
        #
        #         fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        #         ## show prediction active patches
        #         ax1 = plt.subplot(2, 2, 1)
        #         ax1.set_title('Original image', {'fontsize': 8})
        #
        #         img = plt.imread(image_path+string_dir_path)
        #         ax1.imshow(img, 'bone')
        #
        #         ## PREDICTION
        #         ax2 = plt.subplot(2, 2, 2)
        #         ax2.set_title('Predictions: '+ single_image_df.columns.values[i], {'fontsize': 8})
        #         im2 = ax2.imshow(prediction[0, :, :, i-1], 'BuPu')
        #         fig.colorbar(im2, ax=ax2, norm=0)
        #         ax2.set_xlabel("Image prediction : "+str(img_prob_np[0, i-1]))
        #
        #         ## LABELS
        #         ax3 = plt.subplot(2, 2, 3)
        #         ax3.set_title('Labels: ' + single_image_df.columns.values[i], {'fontsize': 8})
        #         ax3.set_xlabel("Image label: "+str(class_label_ground) + str(img_label_np[0, i-1]) +" Bbox available: " + str(has_bbox))
        #         im3 = ax3.imshow(g)
        #         fig.colorbar(im3, ax=ax3, norm=0)
        #
        #         ## BBOX of prediction and label
        #         ax4 = plt.subplot(2, 2, 4)
        #         ax4.set_title('Bounding boxes', {'fontsize': 8})
        #
        #         y = (np.where(g == g.max()))[0]
        #         x = (np.where(g == g.max()))[1]
        #
        #         upper_left_x = np.min(x)
        #         width = np.amax(x)-upper_left_x + 1
        #         upper_left_y = np.amin(y)
        #         height = np.amax(y)-upper_left_y+1
        #         # todo: to draw using pyplot
        #         img4_labels = cv2.rectangle(img, (upper_left_x*64, upper_left_y*64), ((np.amax(x)+1)*64, (np.amax(y)+1)*64), (0, 255, 0), 5)
        #         img4_labels = cv2.rectangle(img, (upper_left_x*64, upper_left_y*64), ((np.amax(x)+1)*64, (np.amax(y)+1)*64), (0, 255, 0), 5)
        #         ax4.imshow(img, 'bone')
        #         # ax4.imshow(img4_labels, 'GnBu')
        #         pred_resized = np.kron(prediction[0, :, :, i-1], np.ones((64,64), dtype=float))
        #         img4_mask = ax4.imshow(pred_resized, 'BuPu', zorder=0, alpha=0.4)
        #
        #
        #         fig.text(0, 0, " Image prediction : "+str(img_prob_np[0, i-1]) + '\n image label: '+
        #                  str(img_label_np[0, i-1]) + '\n IoU: '+ str(iou_score) +
        #                  '\n accuracy: '+ str(acc_per_class_sc),  horizontalalignment='center',
        #                  verticalalignment='center', fontsize=9)
        #
        #         plt.tight_layout()
        #         fig.savefig(results_path + '/images/'+ img_ind+'_'+single_image_df.columns.values[i]+'.jpg', bbox_inches='tight')
        #

