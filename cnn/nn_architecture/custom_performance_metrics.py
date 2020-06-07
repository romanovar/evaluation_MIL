import keras as K
import numpy as np
import tensorflow as tf

from cnn.nn_architecture.custom_loss import compute_ground_truth, compute_image_label_prediction, \
    compute_image_label_in_classification_NORM


def convert_predictions_to_binary(preds, thres):
    return tf.cast(tf.greater_equal(preds, thres), tf.float32)


def reshape_and_convert_to_binary_predictions(predictions, labels, P, threshold_binary, class_nr):
    patches_binary_pred = tf.reshape(convert_predictions_to_binary(predictions, thres=threshold_binary),
                                     (-1, P * P, class_nr))
    return patches_binary_pred, tf.reshape(labels, (-1, P * P, class_nr))


def compute_IoU(predictions, labels, P, class_nr):
    patches_binary_pred, labels_xy_flatten = reshape_and_convert_to_binary_predictions(predictions, labels, P,
                                                                                       threshold_binary=0.5,
                                                                                       class_nr=class_nr)

    correct_prediction = tf.cast(tf.equal(patches_binary_pred, labels_xy_flatten), tf.float32)
    # check only active patches from the labels and see if the prediction there agrees with the labels
    intersection = tf.reduce_sum(
        tf.where(tf.greater(labels_xy_flatten, 0), tf.reshape(correct_prediction, (-1, P * P, class_nr)),
                 tf.zeros((tf.shape(labels_xy_flatten)))), 1)

    union = tf.reduce_sum(patches_binary_pred, 1) + tf.reduce_sum(labels_xy_flatten, 1) - intersection

    return intersection / union


def compute_accuracy_image_bbox(predictions, labels, class_ground_truth, P, iou_threshold, class_nr):
    IoU = compute_IoU(predictions, labels, P, class_nr)
    image_class_pred = tf.cast(tf.greater_equal(IoU, iou_threshold), tf.float32)
    correct_prediction = tf.equal(image_class_pred, class_ground_truth)
    return IoU, tf.cast(correct_prediction, "float")


def compute_accuracy_keras(predictions, instance_labels_ground, P, iou_threshold, class_nr):
    """
    Computes the image probability as loss. Image probability calculation is different when computing in the loss
    function and in the testing conditions ( production). Difference is only for images with available segmentation labels.
    Image probability of images with segmentation in the loss are calculated according by Eq.(1) in
    https://arxiv.org/pdf/1711.06373.pdf
    Image probability of images with NO segmentation in testing/production environment are calculated by Eq.(2) in
    https://arxiv.org/pdf/1711.06373.pdf
    :param y_true: patch labels
    :param y_pred: patch predictions
    :return: Accuracy of images computed according to the formula used during training. Training includes supervised
    training of images with available segmentation. While in testing all images probabilities are calculated in an weakly
    supervised way.
        """
    m = P * P
    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m, class_nr)
    IoU, accuracy_bbox = compute_accuracy_image_bbox(predictions, instance_labels_ground, class_label_ground, P,
                                                     iou_threshold, class_nr)
    img_pred_norm = compute_image_label_in_classification_NORM(predictions, P, class_nr)
    img_pred_bin = tf.cast(img_pred_norm > 0.5, tf.float32)
    correct_prediction_img = tf.cast(tf.equal(img_pred_bin, class_label_ground), tf.float32)

    accuracy_per_obs_per_class = tf.where(has_bbox, accuracy_bbox, correct_prediction_img)
    accuracy_per_class = tf.reduce_mean(accuracy_per_obs_per_class, 0)

    return accuracy_per_class


def keras_accuracy(y_true, y_pred):
    return compute_accuracy_keras(y_pred, y_true, P=16, iou_threshold=0.1, class_nr=1)


def compute_image_probability_asloss(nn_output, instance_label_ground_truth, P, class_nr):
    '''
    Computes image probability the same way it is computed in the loss
    :param nn_output:
    :param instance_label_ground_truth:
    :param P:
    :return:
    '''
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, P * P,
                                                                                  class_nr)

    img_label_pred = compute_image_label_prediction(has_bbox, nn_output, instance_label_ground_truth, P, class_nr)
    return class_label_ground_truth, img_label_pred


def compute_image_probability_production(nn_output,instance_label_ground_truth, P, class_nr):
    '''
    This method considers patches with prediction above Ts as active. If a patch is active, and then it belongs to the
     localization and then ONLY EQUATION 2 of the paper is used
    :param nn_output: output from the last layers
    :param P: number of patches
    :return: image probability per class computed using the active patches
    '''
    image_probability = compute_image_label_in_classification_NORM(nn_output, P, class_nr)
    _, class_label_ground_truth, _= compute_ground_truth(instance_label_ground_truth, P*P, class_nr)

    return class_label_ground_truth, image_probability


def combine_predictions_each_batch(current_batch, prev_batches_arr, batch_ind):
    if batch_ind == 0:
        return current_batch
    else:
        return np.concatenate((prev_batches_arr, current_batch))


def accuracy_asloss(y_true, y_pred):
    """
    Computes the image probability as loss. Image probability calculation is different when computing in the loss
    function and in the testing conditions ( production). Difference is only for images with available segmentation labels.
    Image probability of images with segmentation in the loss are calculated according by Eq.(1) in
    https://arxiv.org/pdf/1711.06373.pdf
    Image probability of images with segmentation in testing/production environment are calculated by Eq.(2) in
    https://arxiv.org/pdf/1711.06373.pdf
    :param y_true: patch labels
    :param y_pred: patch predictions
    :return: Accuracy of images computed according to the formula used during training. Training includes supervised
    training of images with available segmentation. While in testing all images probabilities are calculated in an weakly
    supervised way.
        """
    class_label_ground_truth, img_label_pred = compute_image_probability_asloss(y_pred, y_true, 16, class_nr=1)
    return K.metrics.binary_accuracy(class_label_ground_truth, img_label_pred)

