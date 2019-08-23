import tensorflow as tf
import numpy as np
import keras as K


## input handles all classes simultaneously
def compute_image_label_from_localization_NORM(nn_output, y_true, P):
    # epsilon = tf.pow(tf.cast(10, tf.float32), -15)

    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P, 14))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, P * P, 14))


    normalized_pos = ((1 - 0.98) * pos_patches) + 0.98
    normalized_neg = ((1 - 0.98) * neg_patches) + 0.98

    # element wise multiplication is used as a boolean mask to separate active from inactive patches
    norm_pos_patches = normalized_pos*tf.reshape(y_true, (-1, P * P, 14))
    norm_neg_patches = normalized_neg*tf.reshape((1 - y_true), (-1, P * P, 14))

    Pi_pos_patches = tf.reduce_prod(tf.where(norm_pos_patches>0.0, norm_pos_patches, tf.fill(tf.shape(norm_pos_patches),1.0)), axis=1)
    Pi_neg_patches = tf.reduce_prod(tf.where(norm_neg_patches>0.0, norm_neg_patches, tf.fill(tf.shape(norm_neg_patches),1.0)), axis=1)

    return tf.multiply(Pi_pos_patches, Pi_neg_patches)

## input handles all classes simultaneously
def compute_image_label_from_localization(nn_output, y_true, P):
    epsilon = tf.pow(tf.cast(10, tf.float32), -15)

    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P, 14))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, P * P, 14))

    #setting the inactive values to 1000, so that the minimum of the active value is taken
    min_pos_values = tf.reduce_min(tf.where(pos_patches>0.0, pos_patches, tf.fill(tf.shape(pos_patches), 1000.0)), axis=1, keepdims=True)
    min_neg_values = tf.reduce_min(tf.where(neg_patches>0.0, neg_patches, tf.fill(tf.shape(neg_patches), 1000.0)), axis=1, keepdims=True)

    # min_pos_values = find_minimum_element_in_class(pos_patches)
    # min_neg_values = find_minimum_element_in_class(neg_patches)

    # ensuring no division by 0 and 0 in divident gives result 0
    div_pos_result = tf.where(tf.greater((pos_patches - min_pos_values), 0.0),
                          (pos_patches - min_pos_values) /
                          (tf.reduce_max(pos_patches, axis=1, keepdims=True) - min_pos_values + epsilon),
                          tf.zeros(tf.shape(pos_patches - min_pos_values)))

    normalized_pos = ((1 - 0.98) * div_pos_result) + 0.98

    div_neg_res = tf.where(tf.greater((neg_patches - min_neg_values), 0.0), (neg_patches - min_neg_values) /
                      (tf.reduce_max(neg_patches, axis=1, keepdims=True) - min_neg_values + epsilon),
                           tf.zeros(tf.shape(neg_patches-min_neg_values)))
    normalized_neg = ((1 - 0.98) * div_neg_res) + 0.98
    # normalized_pos = normalize_patches_per_class(pos_patches, min_pos_values, 0.98, 1.0)
    # normalized_neg = normalize_patches_per_class(neg_patches, min_pos_values, 0.98, 1.0)

    # element wise multiplication is used as a boolean mask to separate active from inactive patches
    norm_pos_patches = normalized_pos*tf.reshape(y_true, (-1, P * P, 14))
    norm_neg_patches = normalized_neg*tf.reshape((1 - y_true), (-1, P * P, 14))

    Pi_pos_patches = tf.reduce_prod(tf.where(norm_pos_patches>0.0, norm_pos_patches, tf.fill(tf.shape(norm_pos_patches),1.0)), axis=1)
    Pi_neg_patches = tf.reduce_prod(tf.where(norm_neg_patches>0.0, norm_neg_patches, tf.fill(tf.shape(norm_neg_patches),1.0)), axis=1)

    return tf.multiply(Pi_pos_patches, Pi_neg_patches)


def compute_image_label_in_classification_NORM(nn_output, P):
    # epsilon = tf.pow(tf.cast(10, tf.float32), -15)
    subtracted_prob = 1 - nn_output
    flat_mat = tf.reshape(subtracted_prob, (-1, P * P, 14))

    normalized_mat = ((1 - 0.98) * flat_mat) + 0.98
    element_product = tf.reduce_prod(normalized_mat, axis=1)
    return (tf.cast(1, tf.float32) - element_product)


def compute_image_label_in_classification(nn_output, P):
    epsilon = tf.pow(tf.cast(10, tf.float32), -15)

    subtracted_prob = 1 - nn_output
    ### KEEP the dimension for observations and for the classes
    flat_mat = tf.reshape(subtracted_prob, (-1, P * P, 14))
    min_val = tf.reduce_min(tf.where(flat_mat > 0.0, flat_mat, tf.fill(tf.shape(flat_mat), 1000.0)),
                                   axis=1, keepdims=True)

    max_values = tf.reduce_max(flat_mat, axis=1, keepdims=True)
    ## Normalization between [a, b]
    ### ( (b-a) (X - MIN(x)))/ (MAX(x) - Min(x)) + a

    div_result = tf.where(tf.greater(flat_mat - min_val, 0.0), ((flat_mat - min_val) /(max_values - min_val + epsilon)), tf.zeros(tf.shape(flat_mat - min_val)))

    normalized_mat = ((1 - 0.98) * div_result) + 0.98
    element_product = tf.reduce_prod(normalized_mat, axis=1)
    return (tf.cast(1, tf.float32) - element_product)


def compute_image_label_prediction(has_bbox, nn_output_class, y_true_class, P):
    prob = tf.where(has_bbox, compute_image_label_from_localization_NORM(nn_output_class, y_true_class, P),
                    compute_image_label_in_classification_NORM(nn_output_class, P))
    return prob


def custom_CE_loss(is_localization, labels, preds):
    L_bbox = tf.constant(5, dtype=tf.float32)
    epsilon = tf.pow(tf.cast(10, tf.float32), -15)

    loss_loc = -(L_bbox * labels * (tf.log(preds + epsilon))) - (
        L_bbox * (1 - labels) * (tf.log(1 - preds + epsilon)))

    loss_classification = - (labels * (tf.log(preds + epsilon))) - (
        (1 - labels) * (tf.log(1 - preds + epsilon)))

    loss_class = tf.where(is_localization, loss_loc, loss_classification)
    return loss_class


def binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.
    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(K.backend.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)


def keras_CE_loss(is_localization, labels, probs):
    L_bbox = tf.constant(5, dtype=tf.float32)

    # loss_classification_keras = tf.keras.backend.binary_crossentropy(labels,probs, from_logits=False)
    # loss_loc_keras = L_bbox*tf.keras.backend.binary_crossentropy(labels,probs, from_logits=False)
    loss_classification_keras = binary_crossentropy(labels, probs, from_logits=False)
    loss_loc_keras = L_bbox * binary_crossentropy(labels, probs)
    loss_class_keras = tf.where(is_localization, loss_loc_keras, loss_classification_keras)
    return loss_class_keras


def compute_ground_truth(instance_labels_gt, m):
    sum_active_patches = tf.reduce_sum(tf.reshape(instance_labels_gt, (-1, m, 14)), axis=1)
    class_label_ground_truth = tf.cast(tf.greater(sum_active_patches, 0), tf.float32)
    has_bbox = tf.logical_and(tf.less(sum_active_patches, m), tf.greater(sum_active_patches, 0))

    return sum_active_patches, class_label_ground_truth, has_bbox


def test_compute_ground_truth_per_class_numpy(instance_labels_gt, m):
    sum_active_patches = np.sum(np.reshape(instance_labels_gt, (-1, m)), axis=1)

    class_label_ground_truth = False
    if sum_active_patches > 0.0:
        class_label_ground_truth = True
    has_bbox = False
    if m > sum_active_patches > 0:
        has_bbox = True

    return sum_active_patches, class_label_ground_truth, has_bbox


def compute_loss(nn_output, instance_label_ground_truth, P):
    m = P * P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m)

    img_label_pred = compute_image_label_prediction(has_bbox, nn_output, instance_label_ground_truth, P)

    loss_classification = custom_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    loss_classification_keras = keras_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)

    return loss_classification, loss_classification_keras, img_label_pred, class_label_ground_truth


#todo: delete - not currently used
# def loss_L2(Y_hat, Y, P, L2_rate=0.01):
#     total_loss, total_loss_class, pred_prob, image_prob = compute_loss(Y_hat, Y, P)
#     # normal_loss = compute_image_label_classification_v2(Y_hat, Y, P)
#
#     reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#     return total_loss + L2_rate * sum(reg_losses), total_loss_class, pred_prob, image_prob


def compute_loss_keras(nn_output, instance_label_ground_truth, P):
    m = P * P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m)

    img_label_pred = compute_image_label_prediction(has_bbox, nn_output, instance_label_ground_truth, P)

    # sanity check
    # loss_classification = custom_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    loss_classification_keras = keras_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    total_loss = tf.reduce_sum(loss_classification_keras)
    return total_loss


def keras_loss(y_true, y_pred):
    return compute_loss_keras(y_pred, y_true, P=16)


def keras_loss_reg(y_true, y_pred):
    loss =  compute_loss_keras(y_pred, y_true, P=16)
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([ tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in vars
                    if 'bias' not in v.name ]) * 0.001
    # reg_l2 = 0.01 * tf.nn.l2_loss(tf.hidden_weights) + 0.01 * tf.nn.l2_loss(out_weights)
    return loss+lossL2