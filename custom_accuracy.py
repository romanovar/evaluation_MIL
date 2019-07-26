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
# def compute_accuracy_on_patch_level(predictions, labels, P):
#     patches_binary_pred = convert_predictions_to_binary(predictions)
#     correct_prediction = tf.equal(patches_binary_pred, labels)
#     return tf.reduce_mean(tf.cast(tf.reshape(correct_prediction, (-1, P*P, 14)), "float"), 1)


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
    print(IoU)
    image_class_pred = tf.cast(tf.greater_equal(IoU, iou_threshold), tf.float32)
    correct_prediction = tf.equal(image_class_pred, class_ground_truth)
    return IoU, tf.cast(correct_prediction, "float")

# def false_negative_predictions_image(predictions, class_ground_truth, P):
#     correct_predictions_bin = compute_accuracy_on_image_level(predictions, class_ground_truth, P)
#     img_class_pred_bin = compute_class_prediction_binary(predictions, P)
#
#     TP_mat = tf.where(tf.equal(img_class_pred_bin, 1.0), correct_predictions_bin, tf.zeros(tf.shape(img_class_pred_bin)))
#     FP_mat = tf.where(tf.equal(img_class_pred_bin, 1.0), 1.0-correct_predictions_bin, tf.zeros(tf.shape(img_class_pred_bin)))
#     TN_mat = tf.where(tf.equal(img_class_pred_bin, 0.0), correct_predictions_bin, tf.zeros(tf.shape(img_class_pred_bin)))
#     FN_mat = tf.where(tf.equal(img_class_pred_bin, 0.0), 1.0-correct_predictions_bin, tf.zeros(tf.shape(img_class_pred_bin)))
#
#     TP = tf.reduce_sum(TP_mat, 1)
#     FP = tf.reduce_sum(FP_mat, 1)
#     TN = tf.reduce_sum(TN_mat, 1)
#     FN = tf.reduce_sum(FN_mat, 1)
#
#     return TP, FP, TN, FN


# def compute_AUC_per_class(TP, FP, TN, FN):
#     TPR = TP/(TP +FN)
#     FPR = FP/(TN + FP)
#     return TPR, FPR


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


def compute_AUC_keras(predictions, instance_labels_ground, P, iou_threshold):
    m=P*P
    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
    # IoU, label_binary_bbox = compute_accuracy_image_bbox(predictions, instance_labels_ground, class_label_ground, P, iou_threshold)
    label_binary_bbox = compute_image_label_from_IoU(predictions, instance_labels_ground, P, iou_threshold)
    img_label_pred = tf.where(has_bbox, label_binary_bbox,
                              tf.reduce_max(tf.reshape(predictions, (-1, P * P, 14)), 1))
    auc = roc_auc_score(class_label_ground, img_label_pred)
    # print(tf.shape(accuracy_per_class))
    # print(tf.shape(accuracy_per_obs_per_class))
    # return tf.reduce_mean(accuracy_per_obs_per_class)
    return auc

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
    # return tf.reduce_mean(accuracy_per_obs_per_class)
    return accuracy_per_class


def compute_accuracy_keras_revisited(predictions, instance_labels_ground, P):
    m=P*P

    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
    img_prob_pred  = compute_image_label_in_classification_NORM(predictions, P)

    img_pred_01 = convert_predictions_to_binary(img_prob_pred, 0.5)
    accuracy_per_obs_per_class = tf.cast(tf.equal(img_pred_01, class_label_ground), tf.float32)

    accuracy_per_class = tf.reduce_mean(accuracy_per_obs_per_class, 0)
    print("acc revisited")
    print( tf.shape(accuracy_per_class))
    print(tf.shape(accuracy_per_obs_per_class))
    # return tf.reduce_mean(accuracy_per_obs_per_class)
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
    print("acc as loss")
    print(tf.shape(accuracy_per_class))
    # print(tf.shape(accuracy_per_obs_per_class))
    # return tf.reduce_mean(accuracy_per_obs_per_class)
    return accuracy_per_class


def keras_accuracy(y_true, y_pred):
    return compute_accuracy_keras(y_pred, y_true, P=16, iou_threshold=0.1)


def keras_accuracy_revisited(y_true, y_pred):
    return compute_accuracy_keras_revisited(y_pred, y_true, P=16)


def keras_accuracy_asloss(y_true, y_pred):
    return compute_accuracy_keras_as_loss(y_pred, y_true, P=16)


def keras_AUC(y_true, y_pred):
    P = 16
    # _, class_ground_truth, _ = compute_ground_truth(y_true, P*P)
    # # auc, update_op = compute_AUC_image_per_class(y_pred, class_ground_truth, P=P)
    # auc = compute_AUC_sklearn(y_pred, class_ground_truth, P)
    return compute_AUC_keras(y_pred, y_true, P, 0.1)


#########################################################
def compute_micro_avg_ROC(fpr, tpr, roc_auc, y_true, y_pred):
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(np.asarray(y_true).ravel(),
                                                              np.asarray(y_pred).ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def compute_roc_curve_all_class( y_test, y_pred, fpr, tpr, roc_auc,n_classes):
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    for i in range(0):
        print("im in the loop")
        print(i)
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_pred[:, i], pos_label=1)
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


def compute_roc_curve(img_label, img_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, roc_auc = compute_roc_curve_all_class(img_label, img_pred, fpr, tpr, roc_auc, 14 )
    fpr, tpr, roc_auc = compute_micro_avg_ROC(fpr, tpr, roc_auc, img_label, img_pred)
    # keras_utils.plot_ROC_curve(14, fpr, tpr, roc_auc, out_dir=)
    return fpr, tpr,roc_auc


def custom_roc_curve(predictions, instance_labels_ground, P, iou_threshold):
    m = P * P
    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
    # IoU, accuracy_bbox = compute_accuracy_image_bbox(predictions, instance_labels_ground, class_label_ground, P,
    #                                                  iou_threshold)
    image_pred_from_bbox = compute_image_label_from_IoU(predictions, instance_labels_ground, P, iou_threshold)
    img_max_prob_nobbox = tf.reduce_max(tf.reshape(predictions, (-1, P * P, 14)), 1)
    class_label_ground_binary = np.array(tf.cast(class_label_ground, tf.int32))
    # tf.Print("tensors:", class_label_ground)
    image_predictions = tf.where(has_bbox, image_pred_from_bbox, img_max_prob_nobbox)
    # fpr, tpr, roc_auc = compute_roc_curve(class_label_ground, image_predictions)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(P):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(image_predictions[:,i], class_label_ground[:,i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(np.asarray(predictions).ravel(), np.asarray(instance_labels_ground).ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
    final = tf.reduce_mean(roc_auc)
    return final


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


def keras_AUC_v2(y_true, y_pred):
    return custom_roc_curve(y_pred, y_true, 16, iou_threshold=0.1)


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