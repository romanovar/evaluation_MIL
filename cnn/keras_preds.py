import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from cnn.nn_architecture.custom_performance_metrics import combine_predictions_each_batch, compute_auc_1class
import cnn.nn_architecture.keras_generators as gen
from cnn.keras_utils import normalize, save_evaluation_results, plot_roc_curve, plot_confusion_matrix


def predict_patch_and_save_results(saved_model, file_unique_name, data_set, processed_y,
                                   test_batch_size, box_size, image_size, res_path, mura_interpolation,
                                   resized_images_before_training):
    test_generator = gen.BatchGenerator(
        instances=data_set.values,
        resized_image = resized_images_before_training,
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


def get_patch_labels_from_batches(generator, path, file_name):
    all_img_ind = []
    all_patch_labels = []
    for batch_ind in range(generator.__len__()):
        x, y = generator.__getitem__(batch_ind)
        y_cast = y.astype(np.float32)
        res_img_ind = generator.get_batch_image_indices(batch_ind)
        all_img_ind = combine_predictions_each_batch(res_img_ind, all_img_ind, batch_ind)
        all_patch_labels = combine_predictions_each_batch(y_cast, all_patch_labels, batch_ind)
    np.save(path + 'image_indices_' + file_name, all_img_ind)
    np.save(path + 'patch_labels_' + file_name, all_patch_labels)
    return all_img_ind, all_patch_labels


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
    return (2 * intersection) / (union + intersection)


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
    normalized_neg_patches = ((1 - 0.98) * (1 - patch_pred)) + 0.98

    # element wise multiplication is used as a boolean mask to separate active from inactive patches
    norm_pos_patches = np.ma.masked_array(normalized_pos_patches, mask=pos_patch_labels_mask)
    norm_neg_patches = np.ma.masked_array(normalized_neg_patches, mask=neg_patch_labels_mask)

    product_positive_patches = np.prod(norm_pos_patches, axis=(1, 2))
    product_negative_patches = np.prod(norm_neg_patches, axis=(1, 2))

    return product_positive_patches * product_negative_patches


def compute_bag_prediction_nor(patch_pred):
    subtracted_prob = 1 - patch_pred

    normalized_mat = ((1 - 0.98) * subtracted_prob) + 0.98
    element_product = np.prod(normalized_mat, axis=(1, 2))
    return (1.0 - element_product)


def save_generated_files(res_path, file_unique_name, image_labels, image_predictions, has_bbox,
                         accurate_localizations, dice):
    np.save(res_path + '/image_labels_' + file_unique_name, image_labels)
    np.save(res_path + '/image_predictions_' + file_unique_name, image_predictions)
    np.save(res_path + '/bbox_present_' + file_unique_name, has_bbox)
    np.save(res_path + '/accurate_localization_' + file_unique_name, accurate_localizations)
    np.save(res_path + '/dice_' + file_unique_name, dice)

import pandas as pd


def save_dice(img_ind,dice, res_path, file_identifier):
    df = pd.DataFrame()
    df['Image_ind'] = 0

    df['DICE'] = -100

    df['Image_ind'] = img_ind
    df['Mean DICE'] = dice
    df.to_csv(res_path + 'dice_inst_' + file_identifier + '.csv')


def process_prediction(file_unique_name, res_path, pool_method, img_pred_method, r,
                       threshold_binarization=0.5, iou_threshold=0.1):
    '''
       Processes prediction on bag and instance level. For bag level - bag prediction is computed, for instance level:
    iou and accuracy from iou
    :param file_unique_name: common string of image_indices_.../patch_labels_.../predictions_... .npy files
    :param res_path: path to .npy files
    :param pool_method: mean/ nor / lse
    :param img_pred_method: as_production: is the official prediction method. as_training: is the prediction method
    used during training. It has only supportive function, it is meant to give more insight.
    :param r: R hyperparameter for LSE pooling method
    :param threshold_binarization: binarization threshold of the predictions for iou
    :param iou_threshold: iou threshold for accurate predictions on image level
    :return:
    '''
    predictions, image_indices, patch_labels = get_index_label_prediction(file_unique_name, res_path)
    patch_labels_sum = patch_labels.sum(axis=(1, 2))
    has_bbox = np.greater(patch_labels_sum, 0) & np.less(patch_labels_sum, 256)
    image_labels = np.greater(patch_labels_sum, 0).astype(float)

    image_predictions = compute_bag_prediction(predictions, has_bbox, patch_labels, pool_method=pool_method, r=r,
                                               image_prediction_method=img_pred_method)

    accurate_localization = np.where(has_bbox, compute_accuracy_on_segmentation(predictions, patch_labels,
                                                                                th_binarization=threshold_binarization,
                                                                                th_iou=iou_threshold), 0)
    dice_scores = np.where(has_bbox, compute_dice(predictions, patch_labels, th_binarization=threshold_binarization),
                           -1)
    inst_auc_coll = []
    image_indices_bbox = np.where(dice_scores>-1)[0]
    if len(image_indices_bbox) > 0:
        save_dice(image_indices[image_indices_bbox],dice_scores[image_indices_bbox], res_path, file_unique_name)
    index_segmentaion_images = np.where(has_bbox == True)[0]
    for ind in range(index_segmentaion_images.shape[0]):

        inst_auc = roc_auc_score(patch_labels[index_segmentaion_images[ind]].reshape(-1),
                                 predictions[index_segmentaion_images[ind]].reshape(-1))
        inst_auc_coll.append([inst_auc])
    return image_labels, image_predictions, has_bbox, accurate_localization, dice_scores, inst_auc_coll


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


def compute_save_inst_auc_results(data_set_name, res_path, inst_auc):
    mean_auc = np.mean(inst_auc, axis=0)
    print("Instance AUC")
    print(mean_auc)
    save_evaluation_results(["inst_auc"], mean_auc, "inst_auc_" + data_set_name + '.csv', res_path,
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
                            + image_pred_method + '.csv',
                            res_path)

    plot_roc_curve(fpr, tpr, roc_auc, data_set_name, res_path)
    conf_matrix = confusion_matrix(image_labels, np.array(image_predictions > 0.5, dtype=np.float32))

    plot_confusion_matrix(conf_matrix, [0, 1], res_path, data_set_name, normalize=False, title=None)
    plot_confusion_matrix(conf_matrix, [0, 1], res_path, data_set_name + 'norm', normalize=True, title=None)


def compute_bag_prediction_mean_on_segmentation(nn_output, patch_labels):
    '''
    this function shows the pooling mechanism for annotated images ONLY during TRAINING
    It is implemented for extra insight, but not used to evaluate performance of the algorithm
    It is numpy equivalent of the tensorflow function used during training
    :param nn_output:
    :param patch_labels:
    :return:
    '''
    pos_patch_labels_mask = np.equal(patch_labels, 0.0)
    neg_patch_labels_mask = np.equal(patch_labels, 1.0)

    pos_patches_masked = np.ma.masked_array(nn_output, mask=pos_patch_labels_mask, fill_value=0)
    neg_patches_masked = np.ma.masked_array(1 - nn_output, mask=neg_patch_labels_mask, fill_value=0)

    mean = np.mean(pos_patches_masked.filled() + neg_patches_masked.filled(), axis=(1, 2))
    return mean


def compute_bag_prediction_mean(patch_pred):
    sum_patches = np.sum(patch_pred, axis=(1, 2))
    assert np.mean(patch_pred, axis=(1, 2)).all() == np.sum((1 / 256) * sum_patches, axis=1).all(), "asserion bag error"
    return np.mean(patch_pred, axis=(1, 2))


def compute_bag_prediction_lse(patch_pred, r):
    mean_exp_patches = np.mean(np.exp(r * patch_pred), axis=(1, 2))
    assert ((1 / r) * (np.log(mean_exp_patches))).all() == np.sum((1 / 256) * np.exp(r * patch_pred), axis=1).all(), \
        "asserion bag error"
    return (1 / r) * (np.log(mean_exp_patches))


def compute_bag_prediction_lse_on_segmentation(nn_output, patch_labels, r):
    pos_patch_labels_mask = np.equal(patch_labels, 0.0)
    neg_patch_labels_mask = np.equal(patch_labels, 1.0)
    pos_patches_masked = np.ma.masked_array(nn_output, mask=pos_patch_labels_mask, fill_value=0)
    neg_patches_masked = np.ma.masked_array(1 - nn_output, mask=neg_patch_labels_mask, fill_value=0)

    pos_patches = r * pos_patches_masked
    neg_patches = r * neg_patches_masked

    mean = np.mean((np.exp(pos_patches)).filled() + (np.exp(neg_patches)).filled(), axis=(1, 2))
    result = (1 / r) * np.log(mean)

    sum_pos_patches = np.sum(np.exp(pos_patches).filled(), axis=(1, 2), keepdims=True)
    sum_neg_patches = np.sum(np.exp(neg_patches).filled(), axis=(1, 2), keepdims=True)
    sum_total = sum_neg_patches + sum_pos_patches
    mean2 = np.sum((1 / (256)) * sum_total, axis=(1, 2))
    result2 = (1 / r) * np.log(mean2)
    assert (result == result2).all(), "error in lse computation"

    return result


def compute_bag_prediction_max(patch_pred):
    return np.max(patch_pred, axis=(1, 2))


def compute_bag_prediction_as_production(patch_pred, pool_method, lse_r):
    '''

    :param patch_pred:
    :param pool_method:
    :param lse_r: R hyperparameter - only applicable if the pooling method is 'LSE'
    :return:
    '''
    assert pool_method in ['mean', 'nor', 'lse', 'max'], "ensure you have the right pooling method "
    if pool_method.lower() == 'nor':
        return compute_bag_prediction_nor(patch_pred)
    elif pool_method.lower() == 'mean':
        return compute_bag_prediction_mean(patch_pred)
    elif pool_method.lower() == 'lse':
        return compute_bag_prediction_lse(patch_pred, r=lse_r)
    elif pool_method.lower() == 'max':
        return compute_bag_prediction_max(patch_pred)


def compute_bag_prediction_as_training(has_bbox, predictions, patch_labels, pool_method,r):
    '''
    Calculates the image prediction the way it is computed during training.
    That means that predictions on images with annotations are computed with supervised pooling method
    :param has_bbox: image has annotation
    :param predictions: raws predictions on the image
    :param patch_labels: patch annotation
    :param pool_method: pooling method
    :return:
    '''
    if pool_method.lower() == 'nor':
        return np.where(has_bbox,
                        compute_bag_prediction_nor_on_segmentation(nn_output=predictions, patch_labels=patch_labels),
                        compute_bag_prediction_nor(predictions))
    elif pool_method.lower() == 'mean':
        return np.where(has_bbox,
                        compute_bag_prediction_mean_on_segmentation(nn_output=predictions, patch_labels=patch_labels),
                        compute_bag_prediction_mean(predictions))
    elif pool_method.lower() == 'lse':
        return np.where(has_bbox,
                        compute_bag_prediction_lse_on_segmentation(nn_output=predictions, patch_labels=patch_labels, r=r),
                        compute_bag_prediction_lse(predictions, r))


def compute_bag_prediction(predictions, has_bbox, patch_labels, pool_method, r, image_prediction_method='as_production'):
    '''
    Calculates the bag prediction according to the specified pool method and image prediction method.
    :param predictions:
    :param has_bbox:
    :param patch_labels:
    :param pool_method:
    :param image_prediction_method:
    :param r: R hyperparameter only applicable is pooling method is 'LSE'
    :return:
    '''
    assert image_prediction_method.lower() in ['as_production', 'as_training'], "Invalid image prediction method"
    assert pool_method in ['mean', 'nor', 'lse', 'max'], "ensure you have the right pooling method "

    if image_prediction_method.lower() == 'as_production':
        return compute_bag_prediction_as_production(predictions, pool_method, r)
    elif image_prediction_method.lower() == 'as_training':
        return compute_bag_prediction_as_training(has_bbox, predictions, patch_labels, pool_method, r)
