import sklearn
import tensorflow as tf
import keras_utils
from custom_loss import compute_ground_truth, compute_image_label_prediction, \
    compute_image_label_in_classification_NORM, compute_image_label_from_localization_NORM
import keras as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np
import pandas as pd

from load_data import FINDINGS, PATCH_SIZE


def convert_predictions_to_binary(preds, thres):
    return tf.where(preds > thres, tf.ones(tf.shape(preds)), tf.zeros(tf.shape(preds)))
    tf.cast(tf.greater_equal(preds, thres), tf.float32)


def reshape_and_convert_to_binary_predictions(predictions, labels, P, threshold_binary):
    # predictions_xy_flatten = tf.reshape(predictions, (-1, P * P, 14))
    # labels_xy_flatten = tf.reshape(labels, (-1, P * P, 14))

    patches_binary_pred = tf.reshape(convert_predictions_to_binary(predictions, thres=threshold_binary), (-1, P * P, 14))
    return patches_binary_pred, tf.reshape(labels, (-1, P * P, 14))


def compute_IoU(predictions, labels, P):
    # predictions_xy_flatten = tf.reshape(predictions, (-1, P*P, 14))
    # labels_xy_flatten = tf.reshape(labels, (-1, P*P, 14))
    #
    # patches_binary_pred = convert_predictions_to_binary(predictions, thres=0.5)
    patches_binary_pred, labels_xy_flatten = reshape_and_convert_to_binary_predictions(predictions, labels, P, threshold_binary=0.5)

    correct_prediction = tf.cast(tf.equal(patches_binary_pred, labels_xy_flatten), tf.float32)
    #check only active patches from the labels and see if the prediction there agrees with the labels
    intersection = tf.reduce_sum(tf.where(tf.greater(labels_xy_flatten, 0), tf.reshape(correct_prediction, (-1, P*P, 14)), tf.zeros((tf.shape(labels_xy_flatten)))), 1)

    union = tf.reduce_sum(patches_binary_pred, 1) + tf.reduce_sum(labels_xy_flatten, 1) - intersection

    return intersection/union


# def compute_image_label_from_IoU(predictions, labels, P, iou_thres):
#     IoU = compute_IoU(predictions, labels, P)
#     image_class_pred = tf.cast(tf.greater_equal(IoU, iou_thres), tf.float32)
#     return image_class_pred


def compute_accuracy_image_bbox(predictions, labels, class_ground_truth, P, iou_threshold):
    IoU = compute_IoU(predictions, labels, P)
    image_class_pred = tf.cast(tf.greater_equal(IoU, iou_threshold), tf.float32)
    correct_prediction = tf.equal(image_class_pred, class_ground_truth)
    return IoU, tf.cast(correct_prediction, "float")

#
#
# def compute_AUC_image_per_class(predictions, class_ground_truth, P):
#         max_prediction_class = tf.reduce_max(tf.reshape(predictions, (-1, P*P, 14)), 1)
#         auc = tf.metrics.auc(class_ground_truth, max_prediction_class)
#         tf.keras.backend.get_session().run(tf.local_variables_initializer())
#         return auc


# def compute_AUC_sklearn(predictions, class_ground_truth, P):
#     max_prediction = tf.reduce_max(tf.reshape(predictions, (-1, P * P, 14)), 1)
#     score = tf.py_func(
#             lambda class_ground_truth, max_prediction: roc_auc_score(class_ground_truth, max_prediction, average='macro', sample_weight=None).astype('float32'),
#             [class_ground_truth, max_prediction],
#             'float32',
#             stateful=False,
#             name='sklearnAUC')
#     return score


def compute_class_prediction_binary(predictions, P):
    # img_class_prob_pred = compute_image_label_in_classification(predictions, P)
    active_patches = tf.cast(predictions > 0.5, tf.float32)
    sum_active_patches = tf.reduce_sum(tf.reshape(active_patches, (-1, P * P, 14)), 1)
    img_class_pred_bin = tf.cast(tf.greater_equal(sum_active_patches, 1), tf.float32)
    return img_class_pred_bin


def compute_accuracy_on_image_level(predictions, class_ground_truth, P):
    img_class_pred_bin = compute_class_prediction_binary(predictions, P)
    correct_prediction = tf.equal(img_class_pred_bin, class_ground_truth)
    return tf.cast(correct_prediction, "float")


# USES IoU which is not needed
# EVEN if the evaluation is not used, it is needed for compiling the model
def compute_accuracy_keras(predictions, instance_labels_ground, P, iou_threshold):
    m=P*P
    # n_K = tf.reduce_sum(tf.reshape(instance_labels_ground, (-1, P * P, 14)), axis=1)
    # is_localization = tf.logical_and(tf.less(n_K, P * P), tf.greater(n_K, 0))
    # class_label_ground = tf.cast(tf.greater(n_K, 0), tf.float32)
    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
    IoU, accuracy_bbox = compute_accuracy_image_bbox(predictions, instance_labels_ground, class_label_ground, P, iou_threshold)

    accuracy_per_obs_per_class = tf.where(has_bbox, accuracy_bbox,
                        compute_accuracy_on_image_level(predictions, class_label_ground, P))
    accuracy_per_class = tf.reduce_mean(accuracy_per_obs_per_class, 0)
    print(tf.shape(accuracy_per_class))
    # print(tf.shape(accuracy_per_obs_per_class))
    return accuracy_per_class


# def compute_accuracy_keras_revisited(predictions, instance_labels_ground, P):
#     m=P*P
#
#     sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
#     img_prob_pred  = compute_image_label_in_classification_NORM(predictions, P)
#
#     img_pred_01 = convert_predictions_to_binary(img_prob_pred, 0.5)
#     accuracy_per_obs_per_class = tf.cast(tf.equal(img_pred_01, class_label_ground), tf.float32)
#
#     accuracy_per_class = tf.reduce_mean(accuracy_per_obs_per_class, 0)
#     # print("acc revisited")
#     # print( tf.shape(accuracy_per_class))
#     # print(tf.shape(accuracy_per_obs_per_class))
#     return accuracy_per_class


# # this function has evaluation purpose
# ## NOT MEANT to be used as evaluation of the function
# def compute_accuracy_keras_as_loss(predictions, instance_labels_ground, P):
#     m = P * P
#
#     sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
#     img_prob_pred = compute_image_label_prediction(has_bbox, predictions, instance_labels_ground, P)
#
#     img_pred_01 = convert_predictions_to_binary(img_prob_pred, 0.5)
#     accuracy_per_obs_per_class = tf.cast(tf.equal(img_pred_01, class_label_ground), tf.float32)
#
#     accuracy_per_class = tf.reduce_mean(accuracy_per_obs_per_class, 0)
#
#     return accuracy_per_class


def keras_accuracy(y_true, y_pred):
    return compute_accuracy_keras(y_pred, y_true, P=16, iou_threshold=0.1)

#
# def keras_accuracy_revisited(y_true, y_pred):
#     return compute_accuracy_keras_revisited(y_pred, y_true, P=16)
#
#
# def keras_accuracy_asloss(y_true, y_pred):
#     return compute_accuracy_keras_as_loss(y_pred, y_true, P=16)


def accuracy_bbox_IOU(y_pred, instance_labels_ground, P, iou_threshold):
    _, _, has_bbox = compute_ground_truth(instance_labels_ground, P * P)
    iou_scores = tf.where(has_bbox, compute_IoU(y_pred, instance_labels_ground, P), tf.zeros(tf.shape(has_bbox)))

    image_label_pred = tf.cast(tf.greater_equal(iou_scores, iou_threshold), tf.float32)

    # compare image_label prediction and has_bbox
    # tf equal will NOT be a good idea, as 0 in has bbox means absence of bbox and shouldnt be comuted in accuracy
    acc_pred = tf.reduce_sum(image_label_pred, axis=0)
    true_labels = tf.reduce_sum(tf.cast(has_bbox, tf.float32), axis=0)

    acc_per_class = tf.where(tf.greater(true_labels, 0), acc_pred / true_labels, tf.zeros(tf.shape(true_labels)))
    return acc_per_class


def acc_atelectasis(y_true, y_pred):
    return accuracy_bbox_IOU(y_pred, y_true, P=16, iou_threshold=0.1)[0]
    # return acc_all_classes[0]


def acc_cardiomegaly(y_true, y_pred):
    return accuracy_bbox_IOU(y_pred, y_true, P=16, iou_threshold=0.1)[1]


def acc_effusion(y_true, y_pred):
    return accuracy_bbox_IOU(y_pred, y_true, P=16, iou_threshold=0.1)[4]


def acc_infiltration(y_true, y_pred):
    return accuracy_bbox_IOU(y_pred, y_true, P=16, iou_threshold=0.1)[8]


def acc_mass(y_true, y_pred):
    return accuracy_bbox_IOU(y_pred, y_true, P=16, iou_threshold=0.1)[9]


def acc_nodule(y_true, y_pred):
    return accuracy_bbox_IOU(y_pred, y_true, P=16, iou_threshold=0.1)[10]


def acc_pneumonia(y_true, y_pred):
    return accuracy_bbox_IOU(y_pred, y_true, P=16, iou_threshold=0.1)[12]


def acc_pneumothorax(y_true, y_pred):
    return accuracy_bbox_IOU(y_pred, y_true, P=16, iou_threshold=0.1)[13]


def acc_average(y_true, y_pred):
    avg = [acc_atelectasis(y_true, y_pred), acc_cardiomegaly(y_true, y_pred)]
    return tf.reduce_mean(avg)


def test_function_acc_class(y_pred, instance_labels_ground, P, iou_threshold):
    _, _, has_bbox = compute_ground_truth(instance_labels_ground, P * P)
    iou_scores = tf.where(has_bbox, compute_IoU(y_pred, instance_labels_ground, P), tf.zeros(tf.shape(has_bbox)))
    image_label_pred = tf.cast(tf.greater_equal(iou_scores, iou_threshold), tf.float32)

    # compare image_label prediction and has_bbox
    # tf equal will NOT be a good idea, as 0 in has bbox means absence of bbox and shouldnt be comuted in accuracy
    acc_pred = tf.reduce_sum(image_label_pred, axis=0)
    true_labels = tf.reduce_sum(tf.cast(has_bbox, tf.float32), axis=0)

    acc_per_class = tf.where(tf.greater(true_labels, 0), acc_pred / true_labels, tf.zeros(tf.shape(true_labels))) # tf.constant(-1.0, shape=(tf.shape(true_labels)))
    return  has_bbox, true_labels, acc_pred, acc_per_class #, tf.reduce_mean(acc_per_class)

######################################################### AUC ###########################################
#
#
# def auc_score_tf(img_label, img_pred):
#     auc, update_op = tf.metrics.auc(img_label, img_pred)
#     # K.backend.get_session().run(tf.local_variables_initializer())
#     # K.backend.get_session().run(tf.initialize_all_variables())
#     #
#     # K.backend.get_session().run(tf.global_variables_initializer())
#     return auc
#
#
# def auc_score_sklearn(img_label, img_prob_pred):
#     fpr, tpr, thresholds = sklearn.metrics.roc_curve(img_label, img_prob_pred, pos_label=1.0)
#     return sklearn.metrics.auc(fpr, tpr)
#
#
# def compute_auc_tf_as_loss(predictions, instance_labels_ground, P=16):
#     m = P * P
#     sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
#     img_prob_pred = compute_image_label_prediction(has_bbox, predictions, instance_labels_ground, P)
#
#     auc_all_class = []
#     for clas_ind in range(0,14):
#         print("auc")
#         print(class_label_ground[-1, clas_ind])
#         auc = auc_score_tf(class_label_ground[-1, clas_ind], img_prob_pred[-1, clas_ind])
#         auc_all_class.append(auc)
#     return auc_all_class
#
#
# def compute_auc_sklearn_as_loss(predictions, instance_labels_ground, P):
#     m=P*P
#     sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
#     auc = auc_score_sklearn(predictions, class_label_ground)
#
#     # auc = roc_auc_score(class_label_ground, img_label_pred)
#
#     return auc
#
#
# def custom_auc_score(predictions, instance_labels_ground, P, iou_threshold):
#     m = P * P
#     sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
#
#     image_pred_from_bbox = compute_image_label_from_IoU(predictions, instance_labels_ground, P, iou_threshold)
#     img_max_prob_nobbox = tf.reduce_max(tf.reshape(predictions, (-1, P * P, 14)), 1)
#     # class_label_ground_binary = np.array(tf.cast(class_label_ground, tf.int32))
#     image_predictions = tf.where(has_bbox, image_pred_from_bbox, img_max_prob_nobbox)
#
#     # for i in range(P):
#     #     fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(image_predictions[:, i], class_label_ground[:, i])
#     #     roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
#     #
#
#     return class_label_ground,  image_predictions
#
#
# def keras_auc_score(y_true, y_pred):
#     img_label, img_pred = custom_auc_score(y_pred, y_true, 16, iou_threshold=0.1)
#     auc, update_op = tf.metrics.auc(img_label, img_pred)
#     K.backend.get_session().run(tf.local_variables_initializer())
#     return auc
#
# # TODO: NOT USED - remove
# # def keras_AUC_v2(y_true, y_pred):
# #     return custom_roc_curve(y_pred, y_true, 16, iou_threshold=0.1)
#
#
# def keras_AUC(y_true, y_pred):
#     auc =  compute_auc_tf_as_loss(y_pred, y_true)
#     return auc[0], auc[1], auc[2], auc[3], auc[4], auc[5], auc[6], auc[7], auc[8], \
#            auc[9],auc[10], auc[11], auc[12], auc[13]
#
#

#
# as loss

def image_prob_active_patches(nn_output, P):
    detected_active_patches = tf.cast(tf.greater(nn_output, 0.5), tf.float32)
    sum_detected_actgive_patches, _, detected_bbox = compute_ground_truth(detected_active_patches, P*P)
    # img_prob = compute_image_label_from_localization_NORM(nn_output, detected_active_patches, P)
    return compute_image_label_prediction(detected_bbox, nn_output, detected_active_patches, P )


def compute_image_probability_asloss(nn_output, instance_label_ground_truth, P):
    m = P * P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m)

    img_label_pred = compute_image_label_prediction(has_bbox, nn_output, instance_label_ground_truth, P)
    return class_label_ground_truth, img_label_pred


def compute_image_probability_production(nn_output,instance_label_ground_truth, P):
    m = P*P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m)
    img_label_prob = image_prob_active_patches(nn_output, P)
    return class_label_ground_truth, img_label_prob


def compute_image_probability_production_v2(nn_output,instance_label_ground_truth, P):
    _, class_label_ground_truth, _= compute_ground_truth(instance_label_ground_truth, P*P)
    img_label_prob = compute_image_label_in_classification_NORM(nn_output, P)
    return class_label_ground_truth, img_label_prob

##TODO: to fix the range
def compute_auc(labels_all_classes, img_predictions_all_classes):
    auc_all_classes = []
    for ind in range(0,FINDINGS):
        auc_score = roc_auc_score(labels_all_classes[:, ind], img_predictions_all_classes[:, ind])
        auc_all_classes.append(auc_score)
    return auc_all_classes




# def AUC_class3(y_pred, y_true):
#     return compute_auc(y_true, y_pred)[2]

#
# def keras_auc_v3(y_true, y_pred):
#     img_label, img_prob = compute_image_probability(y_pred, y_true, P=16)
#     auc, update_op = tf.metrics.auc(img_label, img_prob)
#     # K.backend.get_session().run(tf.local_variables_initializer())
#     return auc

###################################### HANDLING PREDICTIONS ############################################################

def combine_predictions_each_batch(current_batch, prev_batches_arr, batch_ind):
    if batch_ind==0:
        return np.concatenate((current_batch, ))
    else:
        return np.concatenate((prev_batches_arr, current_batch))



def make_save_predictions(img_name, raw_predictions, image_predictions, image_label, out_dir, file_name):
    predictions_df = pd.DataFrame()
    predictions_df['Dir Path'] = img_name
    predictions_df = predictions_df.reset_index()
    for ind in range(len(FINDINGS)):
        predictions_df[FINDINGS[ind]+'_pred'] = pd.Series([raw_predictions[n, :, :, ind] for n in range(predictions_df.shape[0])])
        predictions_df[FINDINGS[ind]+'_img_pred'] = pd.Series(image_predictions[:, ind])
        predictions_df[FINDINGS[ind]+'_img_label'] = pd.Series(image_label[:, ind])

    predictions_df.to_csv(out_dir+'/' + file_name)


def create_empty_dataset_results():
    predictions_df = pd.DataFrame()
    # df = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])
    predictions_df['Dir Path'] =[]
    predictions_df = predictions_df.reset_index()
    for ind in range(len(FINDINGS)):
        # df = pd.DataFrame(columns=[FINDINGS[ind] + '_pred', FINDINGS[ind] + '_img_pred', FINDINGS[ind] + '_img_label'])
        predictions_df[FINDINGS[ind] + '_pred'] = []
        predictions_df[FINDINGS[ind] + '_img_pred'] = []
        predictions_df[FINDINGS[ind] + '_img_label'] = []
    return predictions_df