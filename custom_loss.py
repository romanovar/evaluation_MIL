import tensorflow as tf
import numpy as np


## input is made per each class
from tensorflow.python.ops.losses.losses_impl import Reduction


def compute_image_label_localization_v1(nn_output, y_true, P):
    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, P * P))

    normalized_pos = ((1 - 0.98) * (pos_patches - tf.reduce_min(pos_patches, axis=1)) /
                      (tf.reduce_max(pos_patches, axis=1) - tf.reduce_min(pos_patches, axis=1))) + 0.98
    normalized_neg = ((1 - 0.98) * (neg_patches - tf.reduce_min(neg_patches, axis=1)) /
                      (tf.reduce_max(neg_patches, axis=1) - tf.reduce_min(neg_patches, axis=1))) + 0.98

    Pi_pos_patches = tf.reduce_prod(normalized_pos, axis=1)
    Pi_neg_patches = tf.reduce_prod(normalized_neg, axis=1)

    return Pi_pos_patches * Pi_neg_patches


def compute_image_label_localization(nn_output, y_true, P):
    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P, 14))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, P * P, 14))

    normalized_pos = ((1 - 0.98) * (pos_patches - tf.reduce_min(pos_patches, axis=1)) /
                      (tf.reduce_max(pos_patches, axis=1) - tf.reduce_min(pos_patches, axis=1))) + 0.98
    normalized_neg = ((1 - 0.98) * (neg_patches - tf.reduce_min(neg_patches, axis=1)) /
                      (tf.reduce_max(neg_patches, axis=1) - tf.reduce_min(neg_patches, axis=1))) + 0.98

    Pi_pos_patches = tf.reduce_prod(normalized_pos, axis=1)
    Pi_neg_patches = tf.reduce_prod(normalized_neg, axis=1)
    return Pi_pos_patches * Pi_neg_patches

def find_minimum_element_in_class(patches):
    return tf.reduce_min(tf.where(patches > 0.0, patches, tf.fill(tf.shape(patches), 1000.0)), axis=1,
                  keepdims=True)


def normalize_patches_per_class(patches, min_element, min_value, max_value):
    return (((max_value - min_value) * (patches - min_element) /
     (tf.reduce_max(patches, axis=1, keepdims=True) - min_element)) + 0.98)


## input handles all classes simultaneously
def compute_image_label_localization_v2(nn_output, y_true, P):
    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P, 14))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, P * P, 14))

    min_pos_values = tf.reduce_min(tf.where(pos_patches>0.0, pos_patches, tf.fill(tf.shape(pos_patches), 1000.0)), axis=1, keepdims=True)

    min_neg_values = tf.reduce_min(tf.where(neg_patches>0.0, neg_patches, tf.fill(tf.shape(neg_patches), 1000.0)), axis=1, keepdims=True)
    # min_pos_values = find_minimum_element_in_class(pos_patches)
    # min_neg_values = find_minimum_element_in_class(neg_patches)

    normalized_pos = ((1 - 0.98) * (pos_patches - min_pos_values) /
                      (tf.reduce_max(pos_patches, axis=1, keepdims=True) - min_pos_values)) + 0.98
    normalized_neg = ((1 - 0.98) * (neg_patches - min_neg_values) /
                      (tf.reduce_max(neg_patches, axis=1, keepdims=True) - min_neg_values)) + 0.98
    # normalized_pos = normalize_patches_per_class(pos_patches, min_pos_values, 0.98, 1.0)
    # normalized_neg = normalize_patches_per_class(neg_patches, min_pos_values, 0.98, 1.0)

    norm_pos_patches = normalized_pos*tf.reshape(y_true, (-1, P * P, 14))
    norm_neg_patches = normalized_neg*tf.reshape((1 - y_true), (-1, P * P, 14))

    Pi_pos_patches = tf.reduce_prod(tf.where(norm_pos_patches>0.0, norm_pos_patches, tf.fill(tf.shape(norm_pos_patches),1.0)), axis=1)
    Pi_neg_patches = tf.reduce_prod(tf.where(norm_neg_patches>0.0, norm_neg_patches, tf.fill(tf.shape(norm_neg_patches),1.0)), axis=1)

    return tf.multiply(Pi_pos_patches, Pi_neg_patches)


## input handles all classes simultaneously
def compute_image_label_classification_v2(nn_output, P):
    subtracted_prob = 1 - nn_output
    ### KEEP the dimension for observations and for the classes
    flat_mat = tf.reshape(subtracted_prob, (-1, P * P, 14))
    min_val = tf.reduce_min(tf.where(flat_mat > 0.0, flat_mat, tf.fill(tf.shape(flat_mat), 1000.0)),
                                   axis=1, keepdims=True)

    max_values = tf.reduce_max(flat_mat, axis=1, keepdims=True)
    ## Normalization between [a, b]
    ### ( (b-a) (X - MIN(x)))/ (MAX(x) - Min(x)) + a

    normalized_mat = ((1 - 0.98) * (flat_mat - min_val) /
                      (max_values - min_val)) + 0.98

    element_product = tf.reduce_prod(normalized_mat, axis=1)
    return (tf.cast(1, tf.float32) - element_product)


## input is made per each class
def compute_image_label_classification_v1(nn_output, P):
    subtracted_prob = 1 - nn_output
    ### KEEP the dimension for observations and for the classes
    flat_mat = tf.reshape(subtracted_prob, (-1, P * P))
    min_val = tf.reshape(tf.reduce_min(flat_mat, axis=1), (-1, 1))
    max_val = tf.reshape(tf.reduce_max(flat_mat, axis=1), (-1, 1))
    ## Normalization between [a, b]
    ### ( (b-a) (X - MIN(x)))/ (MAX(x) - Min(x)) + a

    normalized_mat = ((1 - 0.98) * (flat_mat - min_val) /
                      (max_val - min_val)) + 0.98

    element_product = tf.reduce_prod(normalized_mat, axis=1)
    # res = (tf.cast(1, tf.float32) - element_product)
    return (1 - element_product)


def compute_image_label_classification(nn_output, P):
    subtracted_prob = 1 - nn_output
    ### KEEP the dimension for observations and for the classes
    flat_mat = tf.reshape(subtracted_prob, (-1, P * P, 14))
    min_val = tf.reshape(tf.reduce_min(flat_mat, axis=1), (-1, 1, 14))
    max_val = tf.reshape(tf.reduce_max(flat_mat, axis=1), (-1, 1, 14))
    ## Normalization between [a, b]
    ### ( (b-a) (X - MIN(x)))/ (MAX(x) - Min(x)) + a

    normalized_mat = ((1 - 0.98) * (flat_mat - min_val) /
                      (max_val - min_val)) + 0.98

    element_product = tf.reduce_prod(normalized_mat, axis=1)
    return (tf.cast(1, tf.float32) - element_product)


def compute_loss_per_image_per_class(comparison, nn_output_class, y_true_class, m, n, P):
    prob = tf.where(comparison, compute_image_label_localization_v2(nn_output_class, y_true_class, P),
                    compute_image_label_classification_v2(nn_output_class, P))
    return prob


def compute_loss_v1(nn_output, y_true, P):
    epsilon = tf.pow(tf.cast(10, tf.float32), -7)
    n_K = tf.reduce_sum(tf.reshape(y_true, (-1, P * P, 14)), axis=1)
    m = P * P
    L_bbox = tf.constant(5, dtype=tf.float32)

    total_loss = 0
    pred_probab = []
    true_probab = []
    total_loss_class = []

    for k in range(14):
        n = n_K[:, k]
        nn_output_class = nn_output[:, :, :, k]
        y_true_class = y_true[:, :, :, k]

        image_true_prob = tf.cast(tf.greater(n, 0), tf.float32)
        true_probab.append(image_true_prob)

        is_localization = tf.logical_and(tf.less(n, m), tf.greater(n, 0))
        prob_class = compute_loss_per_image_per_class(is_localization, nn_output_class, y_true_class, m, n, P)

        loss_loc = -(L_bbox * prob_class * (tf.log(image_true_prob + epsilon))) - (
        L_bbox * (1 - prob_class) * (tf.log(1 - image_true_prob + epsilon)))

        loss_classification = - (prob_class * (tf.log(image_true_prob + epsilon))) - (
        (1 - prob_class) * (tf.log(1 - image_true_prob + epsilon)))

        loss_class = tf.where(is_localization, loss_loc, loss_classification)

        total_loss += loss_class
        pred_probab.append(prob_class)
        total_loss_class.append(loss_class)

    return total_loss, total_loss_class, np.asarray(pred_probab), np.asarray(true_probab)

def compute_CE_loss(is_localization, labels, preds):
    L_bbox = tf.constant(5, dtype=tf.float32)
    epsilon = tf.pow(tf.cast(10, tf.float32), -15)

    loss_loc = -(L_bbox * labels * (tf.log(preds + epsilon))) - (
        L_bbox * (1 - labels) * (tf.log(1 - preds + epsilon)))

    loss_classification = - (labels * (tf.log(preds + epsilon))) - (
        (1 - labels) * (tf.log(1 - preds + epsilon)))

    loss_class = tf.where(is_localization, loss_loc, loss_classification)
    return loss_class


def keras_loss(is_localization, labels, probs):
    L_bbox = tf.constant(5, dtype=tf.float32)

    loss_classification_keras = tf.keras.backend.binary_crossentropy(labels,probs, from_logits=False)
    loss_loc_keras = L_bbox*tf.keras.backend.binary_crossentropy(labels,probs, from_logits=False)
    loss_class_keras = tf.where(is_localization, loss_loc_keras, loss_classification_keras)
    return loss_class_keras


def compute_loss(nn_output, y_true, P):
    n_K = tf.reduce_sum(tf.reshape(y_true, (-1, P * P, 14)), axis=1)
    m = P * P
    L_bbox = tf.constant(5, dtype=tf.float32)

    img_labels = tf.cast(tf.greater(n_K, 0), tf.float32)

    is_localization = tf.logical_and(tf.less(n_K, m), tf.greater(n_K, 0))
    img_pred = compute_loss_per_image_per_class(is_localization, nn_output, y_true, m, n_K, P)

    loss_classification = compute_CE_loss(is_localization, img_labels,img_pred)
    loss_classification_keras = keras_loss(is_localization, img_labels, img_pred)

    # loss_classification_keras = tf.keras.backend.binary_crossentropy(image_true_prob,prob_class, from_logits=False)
    # loss_loc_keras = L_bbox*tf.keras.backend.binary_crossentropy(image_true_prob,prob_class, from_logits=False)
    # loss_class_keras = tf.where(is_localization, loss_loc_keras, loss_classification_keras)

    # total_loss_classes = tf.reduce_sum(loss_class, axis=0)
    # total_loss = tf.reduce_sum(loss_class)

    return loss_classification, loss_classification_keras, img_pred, img_labels


def loss_L2(Y_hat, Y, P, L2_rate=0.01):
    total_loss, total_loss_class, pred_prob, image_prob = compute_loss(Y_hat, Y, P)
    # normal_loss = compute_image_label_classification_v2(Y_hat, Y, P)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return total_loss + L2_rate * sum(reg_losses), total_loss_class, pred_prob, image_prob

