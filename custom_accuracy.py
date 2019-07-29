import sklearn
import tensorflow as tf
import keras_utils
from custom_loss import compute_ground_truth, compute_image_label_prediction, compute_image_label_in_classification_NORM
import keras as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np


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


def compute_image_label_from_IoU(predictions, labels, P, iou_thres):
    IoU = compute_IoU(predictions, labels, P)
    image_class_pred = tf.cast(tf.greater_equal(IoU, iou_thres), tf.float32)
    return image_class_pred


def compute_accuracy_image_bbox(predictions, labels, class_ground_truth, P, iou_threshold):
    IoU = compute_IoU(predictions, labels, P)
    image_class_pred = tf.cast(tf.greater_equal(IoU, iou_threshold), tf.float32)
    correct_prediction = tf.equal(image_class_pred, class_ground_truth)
    return IoU, tf.cast(correct_prediction, "float")



def compute_AUC_image_per_class(predictions, class_ground_truth, P):
        max_prediction_class = tf.reduce_max(tf.reshape(predictions, (-1, P*P, 14)), 1)
        auc = tf.metrics.auc(class_ground_truth, max_prediction_class)
        tf.keras.backend.get_session().run(tf.local_variables_initializer())
        return auc


def compute_AUC_sklearn(predictions, class_ground_truth, P):
    max_prediction = tf.reduce_max(tf.reshape(predictions, (-1, P * P, 14)), 1)
    score = tf.py_func(
            lambda class_ground_truth, max_prediction: roc_auc_score(class_ground_truth, max_prediction, average='macro', sample_weight=None).astype('float32'),
            [class_ground_truth, max_prediction],
            'float32',
            stateful=False,
            name='sklearnAUC')
    return score


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


def compute_accuracy_keras_revisited(predictions, instance_labels_ground, P):
    m=P*P

    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
    img_prob_pred  = compute_image_label_in_classification_NORM(predictions, P)

    img_pred_01 = convert_predictions_to_binary(img_prob_pred, 0.5)
    accuracy_per_obs_per_class = tf.cast(tf.equal(img_pred_01, class_label_ground), tf.float32)

    accuracy_per_class = tf.reduce_mean(accuracy_per_obs_per_class, 0)
    # print("acc revisited")
    # print( tf.shape(accuracy_per_class))
    # print(tf.shape(accuracy_per_obs_per_class))
    return accuracy_per_class


# this function has evaluation purpose
## NOT MEANT to be used as evaluation of the function
def compute_accuracy_keras_as_loss(predictions, instance_labels_ground, P):
    m = P * P

    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
    img_prob_pred = compute_image_label_prediction(has_bbox, predictions, instance_labels_ground, P)

    img_pred_01 = convert_predictions_to_binary(img_prob_pred, 0.5)
    accuracy_per_obs_per_class = tf.cast(tf.equal(img_pred_01, class_label_ground), tf.float32)

    accuracy_per_class = tf.reduce_mean(accuracy_per_obs_per_class, 0)

    return accuracy_per_class


def keras_accuracy(y_true, y_pred):
    return compute_accuracy_keras(y_pred, y_true, P=16, iou_threshold=0.1)


def keras_accuracy_revisited(y_true, y_pred):
    return compute_accuracy_keras_revisited(y_pred, y_true, P=16)


def keras_accuracy_asloss(y_true, y_pred):
    return compute_accuracy_keras_as_loss(y_pred, y_true, P=16)

######################################################### AUC ###########################################


def auc_score_tf(img_label, img_pred):
    auc, update_op = tf.metrics.auc(img_label, img_pred)
    K.backend.get_session().run(tf.local_variables_initializer())
    # K.backend.get_session().run(tf.global_variables_initializer())
    return auc


def auc_score_sklearn(img_label, img_prob_pred):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(img_label, img_prob_pred, pos_label=1.0)
    return sklearn.metrics.auc(fpr, tpr)


def compute_auc_tf_as_loss(predictions, instance_labels_ground, P=16):
    m = P * P
    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
    img_prob_pred = compute_image_label_prediction(has_bbox, predictions, instance_labels_ground, P)

    auc_all_class = []
    for clas_ind in range(0,14):
        print("auc")
        print(class_label_ground[-1, clas_ind])
        auc = auc_score_tf(class_label_ground[-1, clas_ind], img_prob_pred[-1, clas_ind])
        auc_all_class.append(auc)
    return auc_all_class


def compute_auc_sklearn_as_loss(predictions, instance_labels_ground, P):
    m=P*P
    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
    auc = auc_score_sklearn(predictions, class_label_ground)

    # auc = roc_auc_score(class_label_ground, img_label_pred)

    return auc


def custom_auc_score(predictions, instance_labels_ground, P, iou_threshold):
    m = P * P
    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)

    image_pred_from_bbox = compute_image_label_from_IoU(predictions, instance_labels_ground, P, iou_threshold)
    img_max_prob_nobbox = tf.reduce_max(tf.reshape(predictions, (-1, P * P, 14)), 1)
    # class_label_ground_binary = np.array(tf.cast(class_label_ground, tf.int32))
    image_predictions = tf.where(has_bbox, image_pred_from_bbox, img_max_prob_nobbox)

    # for i in range(P):
    #     fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(image_predictions[:, i], class_label_ground[:, i])
    #     roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    #

    return class_label_ground,  image_predictions


def keras_auc_score(y_true, y_pred):
    img_label, img_pred = custom_auc_score(y_pred, y_true, 16, iou_threshold=0.1)
    auc, update_op = tf.metrics.auc(img_label, img_pred)
    K.backend.get_session().run(tf.local_variables_initializer())
    return auc

# TODO: NOT USED - remove
# def keras_AUC_v2(y_true, y_pred):
#     return custom_roc_curve(y_pred, y_true, 16, iou_threshold=0.1)


def keras_AUC(y_true, y_pred):
    auc =  compute_auc_tf_as_loss(y_pred, y_true)
    return auc[0], auc[1], auc[2], auc[3], auc[4], auc[5], auc[6], auc[7], auc[8], \
           auc[9],auc[10], auc[11], auc[12], auc[13]


def AUC_class1(y_pred, y_true):
    return compute_auc_tf_as_loss(y_pred, y_true)[0]


def AUC_class2(y_pred, y_true):
    return compute_auc_tf_as_loss(y_pred, y_true)[1]


def AUC_class3(y_pred, y_true):
    return compute_auc_tf_as_loss(y_pred, y_true)[2]

def compute_image_probability(nn_output, instance_label_ground_truth, P):
    m = P * P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m)

    img_label_pred = compute_image_label_prediction(has_bbox, nn_output, instance_label_ground_truth, P)
    return class_label_ground_truth, img_label_pred


def keras_auc_v3(y_true, y_pred):
    img_label, img_prob = compute_image_probability(y_pred, y_true, P=16)
    auc, update_op = tf.metrics.auc(img_label, img_prob)
    K.backend.get_session().run(tf.local_variables_initializer())
    return auc