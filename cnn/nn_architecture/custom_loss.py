import tensorflow as tf
import numpy as np
import keras as K


## input handles all classes simultaneously
def compute_image_label_from_localization_NORM(nn_output, y_true, P, clas_nr):
    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P, clas_nr))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, P * P, clas_nr))


    normalized_pos = ((1 - 0.98) * pos_patches) + 0.98
    normalized_neg = ((1 - 0.98) * neg_patches) + 0.98

    # element wise multiplication is used as a boolean mask to separate active from inactive patches
    # due to the normalization the inactive patches are also converted t0 0.98
    # so with multiplication - we revert them to 0
    norm_pos_patches = normalized_pos*tf.reshape(y_true, (-1, P * P, clas_nr))
    norm_neg_patches = normalized_neg*tf.reshape((1 - y_true), (-1, P * P, clas_nr))

    Pi_pos_patches = tf.reduce_prod(tf.where(norm_pos_patches>0.0, norm_pos_patches, tf.fill(tf.shape(norm_pos_patches),1.0)), axis=1)
    Pi_neg_patches = tf.reduce_prod(tf.where(norm_neg_patches>0.0, norm_neg_patches, tf.fill(tf.shape(norm_neg_patches),1.0)), axis=1)

    return tf.multiply(Pi_pos_patches, Pi_neg_patches)


def compute_image_label_in_classification_NORM(nn_output, P, clas_nr):
    # epsilon = tf.pow(tf.cast(10, tf.float32), -15)
    print("normalizinng probability")
    subtracted_prob = 1 - nn_output
    flat_mat = tf.reshape(subtracted_prob, (-1, P * P, clas_nr))

    normalized_mat = ((1 - 0.98) * flat_mat) + 0.98
    element_product = tf.reduce_prod(normalized_mat, axis=1)
    return (tf.cast(1, tf.float32) - element_product)


def compute_image_label_prediction(has_bbox, nn_output_class, y_true_class, P, class_nr):
    prob = tf.where(has_bbox, compute_image_label_from_localization_NORM(nn_output_class, y_true_class, P, class_nr),
                    compute_image_label_in_classification_NORM(nn_output_class, P, class_nr))
    return prob


def custom_CE_loss(is_localization, labels, preds):
    L_bbox = tf.constant(5, dtype=tf.float32)
    epsilon = tf.pow(tf.cast(10, tf.float32), -15)

    loss_loc = -(L_bbox * labels * (tf.log(preds + epsilon))) - (
        L_bbox * (1 - labels) * (tf.log(1 - preds + epsilon)))
    batch_weight_pos = 1
    batch_weight_neg = 1

    # to get nr positive labels
    pos_labels = tf.reduce_sum(labels, axis=0)

    # to get nr of neg labels
    neg_labels = tf.reduce_sum(tf.ones(tf.shape(labels)) - labels, axis=0)
    batch_weight_pos = tf.where(tf.greater(pos_labels, 0), (pos_labels+neg_labels)/pos_labels, tf.ones(tf.shape(pos_labels)))
    batch_weight_neg = tf.where(tf.greater(neg_labels, 0), (pos_labels+neg_labels)/neg_labels, tf.ones(tf.shape(neg_labels)))

    loss_classification = - (batch_weight_pos*labels * (tf.log(preds + epsilon))) - (
        batch_weight_neg*(1 - labels) * (tf.log(1 - preds + epsilon)))

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


def compute_ground_truth(instance_labels_gt, m, class_nr):
    sum_active_patches = tf.reduce_sum(tf.reshape(instance_labels_gt, (-1, m, class_nr)), axis=1)
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

#todo: delete if not used
# def compute_loss(nn_output, instance_label_ground_truth, P):
    # m = P * P
    # sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m)
    #
    # img_label_pred = compute_image_label_prediction(has_bbox, nn_output, instance_label_ground_truth, P)
    #
    # loss_classification = custom_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    # loss_classification_keras = keras_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    #
    # return loss_classification, loss_classification_keras, img_label_pred, class_label_ground_truth
    #

#todo: delete - not currently used
# def loss_L2(Y_hat, Y, P, L2_rate=0.01):
#     total_loss, total_loss_class, pred_prob, image_prob = compute_loss(Y_hat, Y, P)
#     # normal_loss = compute_image_label_classification_v2(Y_hat, Y, P)
#
#     reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#     return total_loss + L2_rate * sum(reg_losses), total_loss_class, pred_prob, image_prob


def compute_loss_keras(nn_output, instance_label_ground_truth, P, class_nr):
    m = P * P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m, class_nr)

    img_label_pred = compute_image_label_prediction(has_bbox, nn_output, instance_label_ground_truth, P, class_nr)

    # sanity check
    loss_classification_keras = custom_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    #loss_classification_keras = keras_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    total_loss = tf.reduce_sum(loss_classification_keras)
    return total_loss


def keras_loss(y_true, y_pred):
    return compute_loss_keras(y_pred, y_true, P=16, class_nr=1)


def keras_loss_reg(y_true, y_pred):
    loss =  compute_loss_keras(y_pred, y_true, P=16, class_nr=1)
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([ tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in vars
                    if 'bias' not in v.name ]) * 0.0001
    # reg_l2 = 0.01 * tf.nn.l2_loss(tf.hidden_weights) + 0.01 * tf.nn.l2_loss(out_weights)
    return loss+lossL2


##########################################################
#todo: delete if not used
# def binary_CE_loss_v2(is_loc, labels, probs):
#     factor_loc = tf.constant(5, dtype=tf.float32)
#
#     _epsilon = tf.convert_to_tensor(K.backend.epsilon(), probs.dtype.base_dtype)
#
#     probs = tf.clip_by_value(probs, _epsilon, 1 - _epsilon)
#
#     loss_common = -(labels*tf.log(probs))-((1-labels)*tf.log(1-probs))
#
#     loss_class_keras = tf.where(is_loc, factor_loc*loss_common, loss_common)
#     return loss_class_keras
#


def mean_pooling_segmentation_images(nn_output, y_true, P, clas_nr):
    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P, clas_nr))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, P * P, clas_nr))
    mean = tf.reduce_mean(pos_patches +neg_patches, axis=1)

    sum_pos_patches = tf.reduce_sum(tf.where(pos_patches>0.0, pos_patches, tf.fill(tf.shape(pos_patches),0.0)), axis=1,
                                    keepdims=True)
    sum_neg_patches = tf.reduce_sum(tf.where(neg_patches > 0.0, neg_patches, tf.fill(tf.shape(neg_patches), 0.0)),
                                    axis=1, keepdims=True)
    sum_total = tf.add(sum_neg_patches, sum_pos_patches)
    mean2 = tf.reduce_sum(tf.multiply((1/(P*P)), sum_total), axis=1)
    return mean


def mean_pooling_bag_level(nn_output):
    return tf.reduce_mean(nn_output, axis=[1, 2])


def lse_pooling_bag_level(nn_output, r =1):
    mean = tf.reduce_mean(tf.exp(r*nn_output), axis=[1,2])
    return (1/r)*(tf.log(mean))


def lse_pooling_segmentation_images(nn_output, y_true, P, clas_nr, r =1):

    pos_patch_labels_filter = tf.equal(y_true, 1.0)
    neg_patch_labels_filter = tf.equal(y_true, 0.0)

    pos_patches = r *(nn_output)
    neg_patches = r * (1 - nn_output)

    pos_patches_exp = tf.where(pos_patch_labels_filter, tf.exp(pos_patches),
                               tf.fill(tf.shape(pos_patch_labels_filter),0.0))
    neg_patches_exp = tf.where(neg_patch_labels_filter, tf.exp(neg_patches),
                               tf.fill(tf.shape(pos_patch_labels_filter),0.0))
    pos_neg_combined = pos_patches_exp+neg_patches_exp

    mean3 = tf.reduce_mean(pos_neg_combined, axis=[1,2])
    mean2 = tf.reduce_sum((1 / (256))* pos_neg_combined, axis=[1,2])
    result2 = (1 / r) * tf.log(mean2)
    result3 = tf.log(mean3)/r
    return result2


def compute_image_label_prediction_v2(has_bbox, nn_output_class, y_true_class, P, class_nr, pooling_operator):
    assert pooling_operator in ['mean', 'nor', 'lse'], "ensure you have the right pooling method "

    if pooling_operator.lower()=='nor':
        prob = tf.where(has_bbox, compute_image_label_from_localization_NORM(nn_output_class, y_true_class, P, class_nr),
                        compute_image_label_in_classification_NORM(nn_output_class, P, class_nr))
    elif pooling_operator.lower()=='mean':
        prob = tf.where(has_bbox,
                        mean_pooling_segmentation_images(nn_output_class, y_true_class, P, class_nr),
                        mean_pooling_bag_level(nn_output_class))
    elif pooling_operator.lower()=='lse':
        prob = tf.where(has_bbox,
                        lse_pooling_segmentation_images(nn_output_class, y_true_class, P, class_nr),
                        lse_pooling_bag_level(nn_output_class, r=1))

    return prob


def compute_loss_keras_v2(nn_output, instance_label_ground_truth, P, class_nr, pool_method):
    m = P * P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m,
                                                                                  class_nr)
    img_label_pred = compute_image_label_prediction_v2(has_bbox, nn_output, instance_label_ground_truth, P, class_nr,
                                                       pool_method)

    # sanity check
    # loss_classification_keras = custom_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    loss_classification_keras = keras_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    total_loss = tf.reduce_sum(loss_classification_keras)
    return total_loss


def keras_loss_v2(y_true, y_pred):
    return compute_loss_keras_v2(y_pred, y_true, P=16, class_nr=1, pool_method='nor')
