import tensorflow as tf
import numpy as np


## input is made per each class
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
    print(normalized_neg)

    Pi_pos_patches = tf.reduce_prod(normalized_pos, axis=1)
    Pi_neg_patches = tf.reduce_prod(normalized_neg, axis=1)
    return Pi_pos_patches * Pi_neg_patches


## input handles all classes simultaneously
def compute_image_label_localization_v2(nn_output, y_true, P):
    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P, 14))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, P * P, 14))

    normalized_pos = ((1 - 0.98) * (pos_patches - tf.reduce_min(pos_patches, axis=1, keepdims=True)) /
                      (tf.reduce_max(pos_patches, axis=1, keepdims=True) - tf.reduce_min(pos_patches, axis=1,
                                                                                         keepdims=True))) + 0.98
    normalized_neg = ((1 - 0.98) * (neg_patches - tf.reduce_min(neg_patches, axis=1, keepdims=True)) /
                      (tf.reduce_max(neg_patches, axis=1, keepdims=True) - tf.reduce_min(neg_patches, axis=1,
                                                                                         keepdims=True))) + 0.98

    print(normalized_neg)

    Pi_pos_patches = tf.reduce_prod(normalized_pos, axis=1)
    Pi_neg_patches = tf.reduce_prod(normalized_neg, axis=1)

    return tf.multiply(Pi_pos_patches, Pi_neg_patches)


## input handles all classes simultaneously
def compute_image_label_classification_v2(nn_output, P):
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


def compute_loss(nn_output, y_true, P):
    epsilon = tf.pow(tf.cast(10, tf.float32), -7)
    n_K = tf.reduce_sum(tf.reshape(y_true, (-1, P * P, 14)), axis=1)
    m = P * P
    L_bbox = tf.constant(5, dtype=tf.float32)

    image_true_prob = tf.cast(tf.greater(n_K, 0), tf.float32)

    is_localization = tf.logical_and(tf.less(n_K, m), tf.greater(n_K, 0))
    prob_class = compute_loss_per_image_per_class(is_localization, nn_output, y_true, m, n_K, P)

    loss_loc = -(L_bbox * prob_class * (tf.log(image_true_prob + epsilon))) - (
        L_bbox * (1 - prob_class) * (tf.log(1 - image_true_prob + epsilon)))

    loss_classification = - (prob_class * (tf.log(image_true_prob + epsilon))) - (
        (1 - prob_class) * (tf.log(1 - image_true_prob + epsilon)))

    loss_class = tf.where(is_localization, loss_loc, loss_classification)

    total_loss_classes = tf.reduce_sum(loss_class, axis=0)
    total_loss = tf.reduce_sum(loss_class)

    return total_loss, total_loss_classes, prob_class, image_true_prob


def loss_L2(Y_hat, Y, P, L2_rate=0.01):
    total_loss, total_loss_class, pred_prob, image_prob = compute_loss(Y_hat, Y, P)
    # normal_loss = compute_image_label_classification_v2(Y_hat, Y, P)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return total_loss + L2_rate * sum(reg_losses), total_loss_class, pred_prob, image_prob

