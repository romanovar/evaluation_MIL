import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy


def compute_image_label_from_localization_NORM(nn_output, y_true, P, clas_nr):
    """Aggregates the patch predictions for each image to image level prediction. The formula is defined by Eq. (1) in
        https://arxiv.org/pdf/1711.06373.pdf
    We normalize the predictions on positive patches and the predictions on negative patches to [0.98,1] as described
    in the paper.
    :param nn_output: List of raw patch predictions
    :param y_true: List of patch labels
    :param P: Patch size
    :param clas_nr: number of prediction classes
    :return:  A list of image predictions for each image based on the raw predictions. Aggregation from instance level
    predictions to bag level predictions for images with available segmentation. This is a supervised method only used
    in training, not in testing.
    """
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
    """
    Aggregates the patch predictions for each image to image level prediction. The formula is defined by Eq. (2) in
        https://arxiv.org/pdf/1711.06373.pdf
    We normalize (1-nn_output) to [0.98,1] as described in the paper.
    :param nn_output: List of raw patch predictions
    :param P: Patch size
    :param clas_nr: number of prediciton classes
    :return: A list of image predictions for each image based on the raw predictions. Aggregation from instanse level
    predictions to bag level predictions
    """
    subtracted_prob = 1 - nn_output
    flat_mat = tf.reshape(subtracted_prob, (-1, P * P, clas_nr))

    normalized_mat = ((1 - 0.98) * flat_mat) + 0.98
    element_product = tf.reduce_prod(normalized_mat, axis=1)
    return tf.cast(1, tf.float32) - element_product


def compute_image_label_prediction(has_bbox, nn_output_class, y_true_class, P, class_nr):
    """
    Computes image label prediction according to the image. Images with available segmentation are computed by Eq.(1) in
    https://arxiv.org/pdf/1711.06373.pdf
    Image probability of images with NO segmentation in testing/production environment are calculated by Eq.(2) in
    https://arxiv.org/pdf/1711.06373.pdf
    :param has_bbox: True/false flag if segmenetation is available
    :param nn_output_class: raw patch predictions for this class
    :param y_true_class: instance labels/patch labels for this class
    :param P: Patch size
    :param class_nr: number of prediction classes
    :return: image label prediction. it is a value between [0, 1]
    """
    prob = tf.where(has_bbox, compute_image_label_from_localization_NORM(nn_output_class, y_true_class, P, class_nr),
                    compute_image_label_in_classification_NORM(nn_output_class, P, class_nr))
    return prob


def compute_ground_truth(instance_labels_gt, m, class_nr):
    """
    Computes number of active patches, the image label and segmentation availability on an image from its instance labels
    :param instance_labels_gt: ground truth instance/patch labels
    :param m: total number of patches on an image
    :param class_nr: number of prediction classes
    :return: active patches, the image label and segmentation availability for an image for a list of images
    """
    sum_active_patches = tf.reduce_sum(tf.reshape(instance_labels_gt, (-1, m, class_nr)), axis=1)
    class_label_ground_truth = tf.cast(tf.greater(sum_active_patches, 0), tf.float32)
    has_bbox = tf.logical_and(tf.less(sum_active_patches, m), tf.greater(sum_active_patches, 0))

    return sum_active_patches, class_label_ground_truth, has_bbox


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
    return (1/r)*(tf.math.log(mean))


def max_pooling_bag_level(nn_output):
    return tf.reduce_max(nn_output, axis=[1, 2])


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
    mean2 = tf.reduce_sum((1 / (P*P))* pos_neg_combined, axis=[1,2])
    result2 = (1 / r) * tf.math.log(mean2)
    result3 = tf.math.log(mean3)/r
    return result2


def max_pooling_segmentation_images(nn_output, y_true, P, clas_nr):
    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P, clas_nr))
    max = tf.reduce_max(pos_patches, axis=1)
    return max


def compute_image_label_prediction_v2(has_bbox, nn_output_class, y_true_class, P, class_nr, pooling_operator, r):
    assert pooling_operator in ['mean', 'nor', 'lse', 'max'], "ensure you have the right pooling method "

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
                        lse_pooling_bag_level(nn_output_class, r=r))
    elif pooling_operator.lower()=='max':
        prob = tf.where(has_bbox,
                        max_pooling_segmentation_images(nn_output_class, y_true_class, P, class_nr),
                        max_pooling_bag_level(nn_output_class))

    return prob


def compute_loss_v3(nn_output, instance_label_ground_truth, P, class_nr, pool_method, r, bbox_weight):
    '''
    Computes image prediction and compares it with the image label in a binary cross entropy loss functions
    :param nn_output: Patch predictions
    :param instance_label_ground_truth: patch ground truth
    :param P: number of patches to divide the image into, horizontally and vertically
    :param class_nr: number of classes
    :param pool_method: pooling method to derive image prediction
    :param bbox_weight: weight in loss for samples with localization annotation
    :return: loss value
    '''
    m = P * P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m,
                                                                                  class_nr)
    img_label_pred = compute_image_label_prediction_v2(has_bbox, nn_output, instance_label_ground_truth, P, class_nr,
                                                       pool_method, r)

    loss = tf.where(tf.reshape(has_bbox, (-1,)),
                    bbox_weight * binary_crossentropy(class_label_ground_truth, img_label_pred),
                    binary_crossentropy(class_label_ground_truth, img_label_pred))
    return loss

# def keras_loss_v3(y_true, y_pred):
#     return compute_loss_v3(y_pred, y_true, 16, 1, 'nor', r=1, bbox_weight=5)


def keras_loss_v3_nor(y_true, y_pred):
    return compute_loss_v3(y_pred, y_true, 16, 1, 'nor', r=1, bbox_weight=5)


def keras_loss_v3_lse(y_true, y_pred):
    return compute_loss_v3(y_pred, y_true, 16, 1, 'lse', r=1, bbox_weight=5)


def keras_loss_v3_lse01(y_true, y_pred):
    return compute_loss_v3(y_pred, y_true, 16, 1, 'lse', r=0.1, bbox_weight=5)


def keras_loss_v3_mean(y_true, y_pred):
    return compute_loss_v3(y_pred, y_true, 16, 1, 'mean', r=1, bbox_weight=5)


def keras_loss_v3_max(y_true, y_pred):
    return compute_loss_v3(y_pred, y_true, 16, 1, 'max', r=1, bbox_weight=5)
