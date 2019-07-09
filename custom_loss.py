import tensorflow as tf


def find_minimum_element_in_class(patches):
    return tf.reduce_min(tf.where(patches > 0.0, patches, tf.fill(tf.shape(patches), 1000.0)), axis=1,
                  keepdims=True)


def normalize_patches_per_class(patches, min_element, min_value, max_value):
    return (((max_value - min_value) * (patches - min_element) /
     (tf.reduce_max(patches, axis=1, keepdims=True) - min_element + tf.keras.backend.epsilon)) + 0.98)


## input handles all classes simultaneously
def compute_image_label_from_localization(nn_output, y_true, P):
    epsilon = tf.pow(tf.cast(10, tf.float32), -15)

    pos_patches = tf.reshape((nn_output * y_true), (-1, P * P, 14))
    neg_patches = tf.reshape((1 - nn_output) * (1 - y_true), (-1, P * P, 14))

    min_pos_values = tf.reduce_min(tf.where(pos_patches>0.0, pos_patches, tf.fill(tf.shape(pos_patches), 1000.0)), axis=1, keepdims=True)

    min_neg_values = tf.reduce_min(tf.where(neg_patches>0.0, neg_patches, tf.fill(tf.shape(neg_patches), 1000.0)), axis=1, keepdims=True)
    # min_pos_values = find_minimum_element_in_class(pos_patches)
    # min_neg_values = find_minimum_element_in_class(neg_patches)

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

    norm_pos_patches = normalized_pos*tf.reshape(y_true, (-1, P * P, 14))
    norm_neg_patches = normalized_neg*tf.reshape((1 - y_true), (-1, P * P, 14))

    Pi_pos_patches = tf.reduce_prod(tf.where(norm_pos_patches>0.0, norm_pos_patches, tf.fill(tf.shape(norm_pos_patches),1.0)), axis=1)
    Pi_neg_patches = tf.reduce_prod(tf.where(norm_neg_patches>0.0, norm_neg_patches, tf.fill(tf.shape(norm_neg_patches),1.0)), axis=1)

    return tf.multiply(Pi_pos_patches, Pi_neg_patches)


## input handles all classes simultaneously
# def compute_image_label_classification_v2(nn_output, P):
#     # epsilon = tf.keras.backend.epsilon()
#     #
#     epsilon = tf.pow(tf.cast(10, tf.float32), -15)
#
#     subtracted_prob = 1 - nn_output
#     ### KEEP the dimension for observations and for the classes
#     flat_mat = tf.reshape(subtracted_prob, (-1, P * P, 14))
#     min_val = tf.reduce_min(tf.where(flat_mat > 0.0, flat_mat, tf.fill(tf.shape(flat_mat), 1000.0)),
#                                    axis=1, keepdims=True)
#
#     max_values = tf.reduce_max(flat_mat, axis=1, keepdims=True)
#     ## Normalization between [a, b]
#     ### ( (b-a) (X - MIN(x)))/ (MAX(x) - Min(x)) + a
#
#     normalized_mat = ((1 - 0.98) * (flat_mat - min_val) /
#                       (max_values - min_val + epsilon)) + 0.98
#
#     element_product = tf.reduce_prod(normalized_mat, axis=1)
#     return (tf.cast(1, tf.float32) - element_product)


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


def compute_image_label_prediction(has_bbox, nn_output_class, y_true_class, m, n, P):
    prob = tf.where(has_bbox, compute_image_label_from_localization(nn_output_class, y_true_class, P),
                    compute_image_label_in_classification(nn_output_class, P))
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


def keras_CE_loss(is_localization, labels, probs):
    L_bbox = tf.constant(5, dtype=tf.float32)

    loss_classification_keras = tf.keras.backend.binary_crossentropy(labels,probs, from_logits=False)
    loss_loc_keras = L_bbox*tf.keras.backend.binary_crossentropy(labels,probs, from_logits=False)
    loss_class_keras = tf.where(is_localization, loss_loc_keras, loss_classification_keras)
    return loss_class_keras


def compute_ground_truth(instance_labels_gt, m):
    sum_active_patches = tf.reduce_sum(tf.reshape(instance_labels_gt, (-1, m, 14)), axis=1)
    class_label_ground_truth = tf.cast(tf.greater(sum_active_patches, 0), tf.float32)
    has_bbox = tf.logical_and(tf.less(sum_active_patches, m), tf.greater(sum_active_patches, 0))
    return sum_active_patches, class_label_ground_truth, has_bbox


# def classification_labels(instance_label_ground_truth, nn_output, P):
#     m = P*P
#     sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m)
#     combo_img_pred = compute_loss_per_image_per_class(has_bbox, nn_output, instance_label_ground_truth, m, sum_active_patches, P)
#     class_pred = test_image_label_classification_v2(nn_output, P)
#     # return instance_label_ground_truth, n_K, class_label_ground_truth, has_bbox, img_label_pred, div_result, min_val, max_values, class_pred
#     return class_label_ground_truth, has_bbox, combo_img_pred, class_pred


def compute_loss(nn_output, instance_label_ground_truth, P):
    m = P * P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m)
    # n_K = tf.reduce_sum(tf.reshape(instance_label_ground_truth, (-1, P * P, 14)), axis=1)

    # class_label_ground_truth = tf.cast(tf.greater(n_K, 0), tf.float32)

    # has_bbox = tf.logical_and(tf.less(n_K, m), tf.greater(n_K, 0))
    img_label_pred = compute_image_label_prediction(has_bbox, nn_output, instance_label_ground_truth, m, sum_active_patches, P)

    loss_classification = custom_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    loss_classification_keras = keras_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)

    return loss_classification, loss_classification_keras, img_label_pred, class_label_ground_truth


#todo: delete - not currently used
def loss_L2(Y_hat, Y, P, L2_rate=0.01):
    total_loss, total_loss_class, pred_prob, image_prob = compute_loss(Y_hat, Y, P)
    # normal_loss = compute_image_label_classification_v2(Y_hat, Y, P)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    return total_loss + L2_rate * sum(reg_losses), total_loss_class, pred_prob, image_prob

def compute_loss_keras(nn_output, instance_label_ground_truth, P):
    m = P * P
    sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_ground_truth, m)
    # n_K = tf.reduce_sum(tf.reshape(instance_label_ground_truth, (-1, P * P, 14)), axis=1)

    # class_label_ground_truth = tf.cast(tf.greater(n_K, 0), tf.float32)

    # has_bbox = tf.logical_and(tf.less(n_K, m), tf.greater(n_K, 0))
    img_label_pred = compute_image_label_prediction(has_bbox, nn_output, instance_label_ground_truth, m, sum_active_patches, P)

    loss_classification = custom_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    loss_classification_keras = keras_CE_loss(has_bbox, class_label_ground_truth, img_label_pred)
    total_loss = tf.reduce_sum(loss_classification_keras)
    return total_loss

def keras_loss(y_true, y_pred):
    return compute_loss_keras(y_pred, y_true, P=16)
####################################### Computing accuracy ####################################################


def convert_predictions_to_binary(preds, thres):
    return tf.where(preds > thres, tf.ones(tf.shape(preds)), tf.zeros(tf.shape(preds)))


# def compute_accuracy_on_patch_level(predictions, labels, P):
#     patches_binary_pred = convert_predictions_to_binary(predictions, thres=0.5)
#     correct_prediction = tf.equal(patches_binary_pred, labels)
#     return tf.reduce_mean(tf.cast(tf.reshape(correct_prediction, (-1, P * P, 14)), "float"), 1)

def compute_accuracy_image_bbox(predictions, labels, class_ground_truth, P, iou_threshold):
    IoU = compute_IoU(predictions, labels, P)
    print(IoU)
    image_class_pred = tf.where(tf.greater_equal(IoU, 0.1), tf.ones(tf.shape(IoU)), tf.zeros(tf.shape(IoU)))
    correct_prediction = tf.equal(image_class_pred, class_ground_truth)
    return IoU, tf.cast(correct_prediction, "float")

def compute_accuracy_on_image_level(predictions, class_ground_truth, P):
    img_class_prob_pred = compute_image_label_in_classification(predictions, P)
    img_class_pred_bin = tf.where(img_class_prob_pred > 0.5, tf.ones(tf.shape(img_class_prob_pred)),
                                  tf.zeros(tf.shape(img_class_prob_pred)))
    correct_prediction = tf.equal(img_class_pred_bin, class_ground_truth)
    return tf.cast(correct_prediction, "float")


def compute_IoU(predictions, labels, P):
    predictions_xy_flatten = tf.reshape(predictions, (-1, P*P, 14))
    labels_xy_flatten = tf.reshape(labels, (-1, P*P, 14))

    patches_binary_pred = convert_predictions_to_binary(predictions, thres=0.5)

    correct_prediction = tf.cast(tf.equal(patches_binary_pred, labels), tf.float32)
    #check only active patches from the labels and see if the prediction there agrees with the labels
    intersection = tf.reduce_sum(tf.where(tf.greater(labels_xy_flatten, 0), tf.reshape(correct_prediction, (-1, P*P, 14)), tf.zeros((tf.shape(labels_xy_flatten)))), 1)

    union = tf.reduce_sum(predictions_xy_flatten, 1) + tf.reduce_sum(labels_xy_flatten, 1) - intersection

    return intersection/union

def compute_accuracy(predictions, instance_labels_ground, P, iou_threshold):
    m=P*P
    # n_K = tf.reduce_sum(tf.reshape(instance_labels_ground, (-1, P * P, 14)), axis=1)
    # is_localization = tf.logical_and(tf.less(n_K, P * P), tf.greater(n_K, 0))
    # class_label_ground = tf.cast(tf.greater(n_K, 0), tf.float32)
    sum_active_patches, class_label_ground, has_bbox = compute_ground_truth(instance_labels_ground, m)
    IoU, accuracy_bbox = compute_accuracy_image_bbox(predictions, instance_labels_ground, class_label_ground, P, iou_threshold)

    accuracy_per_obs_per_class = tf.where(has_bbox,
                                          accuracy_bbox,
                                          compute_accuracy_on_image_level(predictions, class_label_ground, P))
    accuracy_per_class = tf.reduce_mean(accuracy_per_obs_per_class, 0)
    return accuracy_per_obs_per_class, accuracy_per_class, tf.reduce_mean(accuracy_per_obs_per_class)



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
    # return tf.reduce_mean(accuracy_per_obs_per_class)
    return accuracy_per_class

def keras_accuracy(y_true, y_pred):
    return compute_accuracy_keras(y_pred, y_true, P=16, iou_threshold=0.1)