import pandas as pd
import yaml
import argparse
import tensorflow as tf
from keras.engine.saving import load_model
import keras_generators as gen
import keras_model
import keras_utils
import os
import load_data as ld
from custom_accuracy import keras_accuracy, compute_image_probability_production, compute_image_probability_asloss, \
    keras_binary_accuracy, accuracy_asloss, accuracy_asproduction
from custom_loss import keras_loss
import numpy as np

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


################## LOAD & COMPILE SAVED MODEL AND MAKE PREDICTIONS ##########################

BATCH_SIZE_TEST = 2
IMAGE_SIZE = 512
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
                                                                                                      do_stats=False,
                                                                                                      res_path = generated_images_path)
df_train=df_train_init
print('Training set: '+ str(df_train_init.shape))
print('Validation set: '+ str(df_val.shape))
print('Localization testing set: '+ str(df_bbox_test.shape))
print('Classification testing set: '+ str(df_class_test.shape))

# df_train = keras_utils.create_overlap_set_bootstrap(df_train_init, 0.9, seed=2)
init_train_idx = df_train['Dir Path'].index.values


model = load_model(trained_models_path+'best_model_single_patient_inbatch-weight8.h5', custom_objects={
        'keras_loss': keras_loss, 'keras_accuracy': keras_accuracy, 'keras_binary_accuracy': keras_binary_accuracy,
        'accuracy_asloss': accuracy_asloss, 'accuracy_asproduction': accuracy_asproduction})
model = keras_model.compile_model(model)


test_set = pd.concat([df_bbox_test, df_class_test])
# test_set = pd.concat([df_train])

test_generator = gen.BatchGenerator(
    instances=test_set.values,
    batch_size=BATCH_SIZE_TEST,
    net_h=IMAGE_SIZE,
    net_w=IMAGE_SIZE,
    box_size=BOX_SIZE,
    norm=keras_utils.normalize,
    processed_y=skip_processing,
    shuffle=False)

predictions = model.predict_generator(test_generator, steps=test_generator.__len__(), workers=1)
predictions_tf = tf.cast(predictions, tf.float32)
predictions_np = predictions.astype(float)
np.save(results_path+'/predictions_XY', predictions)

print("PREDICTION SHAPE")
print(len(predictions))
#res_df = create_empty_dataset_results(len(predictions))


patch_labels_all_batches = []
img_ind_all_batches = []
for batch_ind in range(test_generator.__len__()):
    print("new batch")
    print(batch_ind)
    x, y = test_generator.__getitem__(batch_ind)
    y = tf.cast(y, tf.float32)
    res_img_ind = test_generator.get_batch_image_indices(batch_ind)

    l_bound = batch_ind * BATCH_SIZE_TEST
    r_bound = (batch_ind + 1) * BATCH_SIZE_TEST
    img_labels_v1, img_prob_preds_v1 = compute_image_probability_asloss(predictions_tf[l_bound:r_bound, :, :, :], y,
                                                                        P=16, class_nr=1)
    img_labels_v2, img_prob_preds_v2 = compute_image_probability_production(predictions_tf[l_bound:r_bound, :, :, :],
                                                                            y, P=16, class_nr=1)
    # img_labels_v3, img_prob_preds_v3 = compute_image_probability_production_v2(predictions_tf[l_bound:r_bound, :, :, :], y, P=16)

    col_values = []
    ##### NEW
    test_generator[batch_ind]
    keras_utils.visualize_single_image_all_classes(test_set.iloc[l_bound:r_bound], res_img_ind,
                                                   results_path, predictions_np[l_bound:r_bound],
                                                   img_prob_preds_v2, img_labels_v2, skip_processing)
    #     for i in range(0, BATCH_SIZE_TEST):
    #         col_values = [res_img_ind[i]]
    #
    #         for ind in range(len(ld.FINDINGS)):
    #             find_pred = predictions[l_bound+i, :, :, ind]
    #             col_values.append(find_pred)
    #             # res_df.loc[l_bound + i] = pd.Series(find_pred)
    #
    #             sess = tf.Session()
    #
    #             with sess.as_default():
    #                 fin_img_pred = img_prob_preds_v1[i, ind].eval()
    #                 # res_df.loc[l_bound + i] =fin_img_pred
    #                 fin_img_label = img_labels_v1[i, ind].eval()
    #                 # res_df.loc[l_bound + i] =fin_img_label
    #
    #             print(fin_img_label)
    #             print(type(fin_img_label))
    #
    #             col_values.append(fin_img_pred)
    #             col_values.append(fin_img_label)
    #             print(type(fin_img_pred))
    #             print("***************************")
    #             print(col_values)
    #             # res_df.loc[l_bound+i] = col_values
    #             # res_df.to_csv(results_path+ '/' + 'test.csv')
    #
    # init_op = tf.global_variables_initializer()
    #
    # image_labels_loss =0
    # image_prob_predictions_loss = 0
    # image_labels = 0
    # image_prob_predictions = 0
    # has_bbox_test = 0
    # acc_pred_test=0
    # acc_per_class_test = 0
    #
    # # img_labels_v1, img_prob_preds_v1 = compute_image_probability_asloss(predictions, patch_labels_all_batches, P=16)
    # # img_labels_v2, img_prob_preds_v2 = compute_image_probability_production(predictions, patch_labels_all_batches, P=16)
    # # img_labels_v3, img_prob_preds_v3 = compute_image_probability_production_v2(predictions, patch_labels_all_batches, P=16)
    #
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #
    #     image_labels_loss, image_prob_predictions_loss = img_labels_v1.eval(), img_prob_preds_v1.eval()
    #     image_labels, image_prob_predictions  = img_labels_v2.eval(), img_prob_preds_v2.eval()
    #     # image_labels_v3, image_prob_predictions_v3 = img_labels_v3.eval(), img_prob_preds_v3.eval()
    #

    # make_save_predictions(img_ind_all_batches, predictions, image_prob_predictions_loss, image_labels,
    #                           results_path, 'predictions_loss.csv')
    # make_save_predictions(img_ind_all_batches, predictions, image_prob_predictions, image_labels,
    #                           results_path, 'predictions_production.csv')
    # make_save_predictions(img_ind_all_batches, predictions, image_prob_predictions_v3, image_labels,
    #                           results_path, 'predictions_production_v3.csv')