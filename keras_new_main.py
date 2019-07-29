from pathlib import Path

import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model

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
from custom_accuracy import keras_accuracy, compute_image_probability, keras_auc_v3
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
BATCH_SIZE_TEST = 3
BOX_SIZE = 16

if skip_processing:
    xray_df = ld.load_csv(processed_labels_path)
else:
    label_df = ld.get_classification_labels(classication_labels_path, False)
    processed_df = ld.preprocess_labels(label_df, image_path)
    xray_df = ld.couple_location_labels(localization_labels_path, processed_df, ld.PATCH_SIZE, results_path)

print("Splitting data ...")
init_train_idx, df_train, df_val, df_bbox_test, df_class_test = ld.get_train_test(xray_df, random_state=0, do_stats=True, res_path =results_path)
df_train, df_val, df_bbox_test, df_class_test = ld.get_train_test_strata(xray_df, random_state=0, do_stats=True, res_path=results_path)
# ld.train_test_stratification(xray_df, random_state=0, do_stats=True)

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
                               min_delta=0.0001,
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

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(df_train) // BATCH_SIZE,
        epochs=3,
        validation_data=valid_generator,
        validation_steps=len(df_val) // BATCH_SIZE,
        verbose=2,
        callbacks=[checkpoint, early_stop, lrate]
    )

    keras_utils.plot_train_validation(history.history['keras_accuracy'],
                                  history.history['val_keras_accuracy'],
                                  'model accuracy', 'accuracy', results_path)

    keras_utils.plot_train_validation(history.history['loss'],
                                  history.history['val_loss'],
                                  'model loss', 'loss', results_path)
else:

    model = load_model(results_path+'/trained_model.h5', custom_objects={'keras_loss': keras_loss, 'keras_accuracy':keras_accuracy})

    model_new_eval = keras_model.compile_model_on_load(model)

    # if skip_processing:
    # process labels in the same way as in the batch generators
    if not test_single_image:
        test_bbox_generator = gen.BatchGenerator(
            instances=df_bbox_test.values,
            batch_size=BATCH_SIZE_TEST,
            net_h=IMAGE_SIZE,
            net_w=IMAGE_SIZE,
            box_size=BOX_SIZE,
            # net_crop=IMAGE_SIZE,
            norm=keras_utils.normalize,
            processed_y=skip_processing)

        test_class_generator = gen.BatchGenerator(
            instances=df_class_test.values,
            batch_size=BATCH_SIZE_TEST,
            net_h=IMAGE_SIZE,
            net_w=IMAGE_SIZE,
            box_size=BOX_SIZE,
            # net_crop=IMAGE_SIZE,
            norm=keras_utils.normalize,
            processed_y=skip_processing)

        test_set = pd.concat([df_bbox_test, df_class_test])

        test_generator = gen.BatchGenerator(
            instances=test_set.values,
            batch_size=BATCH_SIZE_TEST,
            net_h=IMAGE_SIZE,
            net_w=IMAGE_SIZE,
            box_size=BOX_SIZE,
            # net_crop=IMAGE_SIZE,
            norm=keras_utils.normalize,
            processed_y=skip_processing)

        print(model.metrics_names)

        # predict performs only predictions predict_onbatch
        # evaluate - does prediction and compute evaluation metrics - test_onbatch
        test_bbox_predictions = model.evaluate_generator(
            test_bbox_generator,
            steps = len(df_bbox_test) // BATCH_SIZE_TEST
        )

        test_classification_predictions = model.evaluate_generator(
            test_class_generator,
            steps=len(df_class_test) // BATCH_SIZE_TEST
        )

        test_predictions = model.evaluate_generator(
            test_generator,
            steps = len(test_set)//BATCH_SIZE_TEST
        )

        print(test_bbox_predictions[0], test_bbox_predictions[1], test_bbox_predictions[2], test_bbox_predictions[3])

        print(test_classification_predictions[0], test_classification_predictions[1], test_classification_predictions[2],
              test_classification_predictions[3])
        print(test_predictions[0], test_predictions[1], test_predictions[2], test_predictions[3])

    else:
        dir_img = "img_5/"
        img_ind = "00000211_041"
        string_dir_path = dir_img+img_ind+".png"
        path_2 = Path(image_path+dir_img+img_ind+".png")
        single_image_df = df_bbox_test[df_bbox_test['Dir Path'] == str(path_2)]

        test_generator = gen.BatchGenerator(
            instances=single_image_df.values,
            batch_size=1,
            net_h=IMAGE_SIZE,
            net_w=IMAGE_SIZE,
            box_size=BOX_SIZE,
            norm=keras_utils.normalize,
            processed_y=skip_processing)

        prediction = model.predict(test_generator)

        test_stats = model.evaluate_generator(test_generator, steps = 1)

        x, y = test_generator.__getitem__(0)
        img_label, img_prob = compute_image_probability(prediction[:, :, :, :], y.astype(np.float32), P=16)
        auc = keras_auc_v3(y.astype(np.float32), prediction)


        img_label_np = 0
        img_prob_np = 0

        # initialize the variable
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            img_label_np =img_label.eval()
            img_prob_np = img_prob.eval()

        for row in single_image_df.values:
            labels_df = []

            for i in range(1, row.shape[0]):  # (15)
                g = ld.process_loaded_labels_tf(row[i])

                sum_active_patches, class_label_ground, has_bbox = test_compute_ground_truth_per_class_numpy(g, 16 * 16)
                print("sum active patches: " + str(sum_active_patches))
                print("class label: " + str(class_label_ground))
                print("Has bbox:" + str(has_bbox))


                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                ## show prediction active patches
                ax1 = plt.subplot(2, 2, 1)
                ax1.set_title('Original image', {'fontsize': 8})

                img = plt.imread(image_path+string_dir_path)
                ax1.imshow(img, 'bone')

                ## PREDICTION
                ax2 = plt.subplot(2, 2, 2)
                ax2.set_title('Predictions: '+ single_image_df.columns.values[i], {'fontsize': 8})
                im2 = ax2.imshow(prediction[0, :, :, i-1], 'BuPu')
                fig.colorbar(im2, ax=ax2, norm=0)
                ax2.set_xlabel("Image prediction : "+str(img_prob_np[0, i-1]))

                ## LABELS
                ax3 = plt.subplot(2, 2, 3)
                ax3.set_title('Labels: ' + single_image_df.columns.values[i], {'fontsize': 8})
                ax3.set_xlabel("Image label: "+str(class_label_ground) + str(img_label_np[0, i-1]) +" Bbox available: " + str(has_bbox))
                im3 = ax3.imshow(g)
                fig.colorbar(im3, ax=ax3, norm=0)

                ## BBOX of prediction and label
                ax4 = plt.subplot(2, 2, 4)
                ax4.set_title('Bounding boxes', {'fontsize': 8})

                y = (np.where(g == g.max()))[0]
                x = (np.where(g == g.max()))[1]

                upper_left_x = np.min(x)
                width = np.amax(x)-upper_left_x + 1
                upper_left_y = np.amin(y)
                height = np.amax(y)-upper_left_y+1
                # todo: to draw using pyplot
                img4_labels = cv2.rectangle(img, (upper_left_x*64, upper_left_y*64), ((np.amax(x)+1)*64, (np.amax(y)+1)*64), (0, 255, 0), 5)
                img4_labels = cv2.rectangle(img, (upper_left_x*64, upper_left_y*64), ((np.amax(x)+1)*64, (np.amax(y)+1)*64), (0, 255, 0), 5)
                ax4.imshow(img, 'bone')
                # ax4.imshow(img4_labels, 'GnBu')
                pred_resized = np.kron(prediction[0, :, :, i-1], np.ones((64,64), dtype=float))
                img4_mask = ax4.imshow(pred_resized, 'BuPu', zorder=0, alpha=0.4)
                # cmap=sns.cubehelix_palette(light=1, as_cmap=True, reverse=True), alpha=0.3)
                # print("no bbx drawn !!!")
                # print(type(img4_mask.make_image()))
                # print(type(img))

                fig.text(0, 0, " Image prediction : "+str(img_prob_np[0, i-1]) + '\n image label: '+
                         str(img_label_np[0, i-1]) + '\n Auc: '+ str(test_stats[0]) +
                         '\n accuracy: '+ str(test_stats[0]),  horizontalalignment='center',
                         verticalalignment='center', fontsize=9)

                plt.tight_layout()
                fig.savefig(results_path + '/images/'+ img_ind+'_'+single_image_df.columns.values[i]+'.jpg', bbox_inches='tight')
                # plt.show()


