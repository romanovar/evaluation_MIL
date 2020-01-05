import numpy as np
from sklearn.metrics import confusion_matrix
from cnn.nn_architecture.custom_performance_metrics import combine_predictions_each_batch, compute_auc_1class
import cnn.nn_architecture.keras_generators as gen
from cnn.keras_utils import normalize, save_evaluation_results, plot_roc_curve, plot_confusion_matrix


def predict_patch_and_save_results(saved_model, file_unique_name, data_set, processed_y,
                                   test_batch_size, box_size, image_size, res_path, mura_interpolation=True):
    test_generator = gen.BatchGenerator(
        instances=data_set.values,
        batch_size=test_batch_size,
        net_h=image_size,
        net_w=image_size,
        box_size=box_size,
        norm=normalize,
        processed_y=processed_y,
        shuffle=False,
        interpolation=mura_interpolation
    )

    predictions = saved_model.predict_generator(test_generator, steps=test_generator.__len__(), workers=1)
    np.save(res_path + 'predictions_' + file_unique_name, predictions)

    all_img_ind = []
    all_patch_labels = []
    for batch_ind in range(test_generator.__len__()):
        x, y = test_generator.__getitem__(batch_ind)
        y_cast = y.astype(np.float32)
        res_img_ind = test_generator.get_batch_image_indices(batch_ind)
        all_img_ind = combine_predictions_each_batch(res_img_ind, all_img_ind, batch_ind)
        all_patch_labels = combine_predictions_each_batch(y_cast, all_patch_labels, batch_ind)
    np.save(res_path + 'image_indices_' + file_unique_name, all_img_ind)
    np.save(res_path + 'patch_labels_' + file_unique_name, all_patch_labels)


#################################################################
def load_npy(file_name, res_path):
    return np.load(res_path + file_name, allow_pickle=True)


def get_index_label_prediction(file_set_name, res_path):
    prediction_file = 'predictions_' + file_set_name + '.npy'
    img_ind_file = 'image_indices_' + file_set_name + '.npy'
    patch_labels_file = 'patch_labels_' + file_set_name + '.npy'

    preds = load_npy(prediction_file, res_path)
    img_indices = load_npy(img_ind_file, res_path)
    patch_labs = load_npy(patch_labels_file, res_path)
    return preds, img_indices, patch_labs


def compute_bag_probability_asloss(nn_output, instance_label_ground_truth, P, has_bbox, class_nr, pooling_operator):
    '''
    Computes image probability the same way it is computed in the loss
    So if there are annotated segmentation, the patch labels are taken into account and
    the bag prediction is computed differently
    :param nn_output:
    :param instance_label_ground_truth:
    :param P:
    :return:
    '''
    if pooling_operator.lower() == 'nor':
        np.where(has_bbox, 2, -1)
    return None


# def compute_bag_prediction(raw_predictions, patch_labs, img_pred_as_loss):
#     if img_pred_as_loss == 'as_loss':
#         img_labels, img_prob_preds_v1 = compute_image_probability_asloss(raw_predictions, patch_labs, P=16, class_nr=1)
#     elif img_pred_as_loss == 'as_production':
#         img_labels, img_prob_preds_v1 = compute_image_probability_production(raw_predictions, patch_labs, P=16,
#                                                                              class_nr=1)
#     return img_labels, img_prob_preds_v1

def compute_intersection_union_patches(predictions, patch_labels, threshold_binarization):
    binary_patch_predictions = np.greater_equal(predictions, threshold_binarization, dtype=float)
    correct_prediction = np.equal(binary_patch_predictions, patch_labels, dtype=float)
    # check only active patches from the labels and see if the prediction there agrees with the labels
    intersection = np.where(np.greater(patch_labels, 0), correct_prediction, 0)
    total_patch_intersection = np.sum(intersection, axis=(1, 2))
    union = np.sum(binary_patch_predictions, axis=(1, 2)) + np.sum(patch_labels, axis=(1, 2)) - total_patch_intersection
    return total_patch_intersection, union


def compute_iou(predictions, patch_labels, threshold_binarization):
    intersection, union = compute_intersection_union_patches(predictions, patch_labels, threshold_binarization)
    return intersection / union


def compute_dice(predictions, patch_labels, th_binarization):
    intersection, union = compute_intersection_union_patches(predictions, patch_labels, th_binarization)
    return (2*intersection) / (union+intersection)


def compute_accuracy_on_segmentation(predictions, patch_labels, th_binarization, th_iou):
    iou_score = compute_iou(predictions, patch_labels, th_binarization)
    return np.greater_equal(iou_score, th_iou, dtype=float)


def compute_bag_prediction_nor_on_segmentation(patch_pred, patch_labels):
    '''
    Computes the bag prediction using NOR pooling on images with annotated segmentation
    NB: THIS function is used only during training, or for sanity check, but never in testing condition
    :param patch_pred: patch predictions
    :param patch_labels: patch labels
    :return: probability of positive bag
    '''
    pos_patch_labels_mask = np.equal(patch_labels, 0.0)
    neg_patch_labels_mask = np.equal(patch_labels, 1.0)

    normalized_pos_patches = ((1 - 0.98) * patch_pred) + 0.98
    normalized_neg_patches = ((1 - 0.98) * (1-patch_pred)) + 0.98

    # element wise multiplication is used as a boolean mask to separate active from inactive patches
    norm_pos_patches = np.ma.masked_array(normalized_pos_patches, mask=pos_patch_labels_mask)
    norm_neg_patches = np.ma.masked_array(normalized_neg_patches, mask=neg_patch_labels_mask)

    product_positive_patches = np.prod(norm_pos_patches, axis=(1,2))
    product_negative_patches = np.prod(norm_neg_patches, axis=(1,2))

    return product_positive_patches*product_negative_patches


def compute_bag_prediction_nor(patch_pred):
    subtracted_prob = 1 - patch_pred

    normalized_mat = ((1 - 0.98) * subtracted_prob) + 0.98
    element_product = np.prod(normalized_mat, axis=(1,2))
    return (1.0 - element_product)


def save_generated_files(res_path, file_unique_name, image_labels, image_predictions, has_bbox,
                         accurate_localizations, dice):
    np.save(res_path + '/image_labels_' + file_unique_name, image_labels)
    np.save(res_path + '/image_predictions_' + file_unique_name, image_predictions)
    np.save(res_path + '/bbox_present_' + file_unique_name, has_bbox)
    np.save(res_path + '/accurate_localization_' + file_unique_name, accurate_localizations)
    np.save(res_path + '/dice_'+ file_unique_name, dice)


def process_prediction(file_unique_name, res_path, img_pred_as_loss, threshold_binarization=0.5, iou_threshold=0.1):
    '''
    Processes prediction on bag and instance level. For bag level - bag prediction is computed, for instance level:
    iou and accuracy from iou
    :param file_unique_name:
    :param res_path:
    :param img_pred_as_loss: 'as_production'
    :return:
    '''
    predictions, image_indices, patch_labels = get_index_label_prediction(file_unique_name, res_path)
    patch_labels_sum = patch_labels.sum(axis=(1, 2))
    image_labels = np.greater(patch_labels_sum, 0).astype(float)
    image_predictions=compute_bag_prediction_nor(predictions)
    has_bbox = np.greater(patch_labels_sum, 0) & np.less(patch_labels_sum, 256)
    accurate_localization = np.where(has_bbox, compute_accuracy_on_segmentation(predictions, patch_labels,
                                                                                th_binarization=threshold_binarization,
                                                                                th_iou=iou_threshold), 0)
    dice_scores = np.where(has_bbox, compute_dice(predictions, patch_labels, th_binarization=threshold_binarization), -1)

    return image_labels, image_predictions, has_bbox, accurate_localization, dice_scores


def compute_save_accuracy_results(data_set_name, res_path, has_bbox, acc_localization):
    print("accuracy bbox present vs accurate")
    total_accurate_segmentations = np.sum(acc_localization, axis=0)
    total_segmentation = np.sum(has_bbox, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_class = total_accurate_segmentations / total_segmentation
    print("ACCURACY RESULTS FROM BBOX")
    print(acc_class)
    save_evaluation_results(["accuracy"], acc_class, "accuracy_" + data_set_name + '.csv', res_path,
                                        add_col=None, add_value=None)


def compute_save_dice_results(data_set_name, res_path, has_bbox, dice_scores):
    dice_score_ma = np.ma.masked_array(dice_scores, mask=np.equal(dice_scores, -1))
    mean_dice = np.mean(dice_score_ma, axis=0)
    print("DICE")
    print(mean_dice)
    save_evaluation_results(["dice"], mean_dice, "dice_" + data_set_name + '.csv', res_path,
                            add_col=None, add_value=None)


def compute_save_auc(data_set_name, image_pred_method, res_path, image_labels, image_predictions, class_name):
    '''
    It computes and saves the results in a file. ROC curve and confusion matrix visualizations are also done here.
    :param data_set_name:
    :param image_pred_method:
    :param res_path:
    :param image_labels:
    :param image_predictions:
    :param class_name:
    :return:
    '''
    auc_all_classes_v1, fpr, tpr, roc_auc = compute_auc_1class(image_labels, image_predictions)
    save_evaluation_results([class_name], auc_all_classes_v1, 'auc_prob_' + data_set_name + '_'
                                        + image_pred_method+ '.csv',
                                        res_path)

    plot_roc_curve(fpr, tpr, roc_auc, data_set_name, res_path)
    conf_matrix = confusion_matrix(image_labels, np.array(image_predictions > 0.5, dtype=np.float32))

    plot_confusion_matrix(conf_matrix, [0, 1], res_path, data_set_name, normalize=False, title=None)
    plot_confusion_matrix(conf_matrix, [0, 1], res_path, data_set_name+'norm', normalize=True, title=None)