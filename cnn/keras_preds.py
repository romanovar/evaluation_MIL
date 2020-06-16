import os

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from cnn.nn_architecture.custom_performance_metrics import combine_predictions_each_batch
import cnn.nn_architecture.keras_generators as gen
from cnn.keras_utils import normalize, save_evaluation_results, plot_roc_curve, plot_confusion_matrix, \
    image_larger_input, calculate_scale_ratio, set_dataset_flag, build_path_results
from pathlib import Path
from keras_preprocessing.image import load_img, img_to_array
from sklearn.metrics import roc_auc_score, roc_curve, auc

from cnn.preprocessor.load_data_mura import padding_needed, pad_image


def predict_patch_and_save_results(saved_model, file_unique_name, data_set, processed_y,
                                   test_batch_size, box_size, image_size, res_path, mura_interpolation,
                                   resized_images_before_training):
    test_generator = gen.BatchGenerator(
        instances=data_set.values,
        resized_image=resized_images_before_training,
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
    # np.save(res_path + '/accurate_localization_' + file_unique_name, accurate_localizations)
    np.save(res_path + '/dice_' + file_unique_name, dice)


import pandas as pd


def save_dice(img_ind, dice, res_path, file_identifier):
    df = pd.DataFrame()
    df['Image_ind'] = 0

    df['DICE'] = -100

    df['Image_ind'] = img_ind
    df['DICE'] = dice
    df.to_csv(os.path.join(res_path, 'dice_inst_' + file_identifier + '.csv'))


def compute_auc_roc_curve(labels, predictions):
    auc_score = roc_auc_score(labels, predictions)
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    return auc_score, fpr, tpr, roc_auc


def compute_auc_1class(labels_all_classes, img_predictions_all_classes):
    auc_all_classes = []
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for ind in range(0, 1):
        auc_score, fpr1, tpr1, roc_auc1 = compute_auc_roc_curve(labels_all_classes[:, ind],
                                                                img_predictions_all_classes[:, ind])
        auc_all_classes.append(auc_score)
        fpr[ind], tpr[ind] = fpr1, tpr1
        roc_auc[ind] = roc_auc1

    return auc_all_classes, fpr, tpr, roc_auc


def process_prediction(config, file_unique_name, res_path, pool_method, img_pred_method, r,
                       threshold_binarization=0.5, iou_threshold=0.1):
    '''
       Processes prediction on bag and instance level. For bag level - bag prediction is computed, for instance level:
    iou and accuracy from iou
    :param file_unique_name: common string of image_indices_.../patch_labels_.../predictions_... .npy files
    :param res_path: path to .npy files
    :param pool_method: mean/ nor / lse
    :param img_pred_method: as_production: is the official prediction method. as_training: is the prediction method
    used during training. It has only supportive function, it is meant to give more insight.
            See differences in compute_image_label_prediction() in custom_loss.py
    :param r: R hyperparameter for LSE pooling method
    :param threshold_binarization: binarization threshold of the predictions for iou
    :param iou_threshold: iou threshold for accurate predictions on image level
    :return:
    '''

    use_xray, use_pascal = set_dataset_flag(config['dataset_name'])
    pascal_img_path = config['pascal_image_path']

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

    if use_pascal:
        has_bbox, accurate_localizations_inst, dice_scores_inst, indices_to_keep = \
            evaluate_instance_performance_pascal(pascal_img_path, file_unique_name, res_path, has_bbox)
        accurate_localization[indices_to_keep] = accurate_localizations_inst
        dice_scores[indices_to_keep] = dice_scores_inst

    image_indices_bbox = np.where(dice_scores > -1)[0]

    if len(image_indices_bbox) > 0:
        performance_path = os.path.join(os.path.abspath(os.path.join(res_path, os.pardir)), 'performance')
        save_dice(image_indices[image_indices_bbox], dice_scores[image_indices_bbox], performance_path, file_unique_name)
    return image_labels, image_predictions, has_bbox, accurate_localization, dice_scores


def compute_save_accuracy_results(eval_df, data_set_name, res_path, has_bbox, acc_localization):
    print("accuracy bbox present vs accurate")
    total_accurate_segmentations = np.sum(acc_localization, axis=0)
    total_segmentation = np.sum(has_bbox, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_class = total_accurate_segmentations / total_segmentation
    print("ACCURACY RESULTS FROM BBOX")
    print(acc_class)
    # save_evaluation_results(["accuracy"], acc_class, "accuracy_" + data_set_name + '.csv', res_path,
    #                         add_col=None, add_value=None)
    return save_evaluation_results(eval_df, ["accuracy"], acc_class, "evaluation_performance_" + data_set_name + '.csv',
                                   res_path, add_col=None, add_value=None)


def compute_save_dice_results(eval_df, data_set_name, res_path, dice_scores):
    dice_score_ma = np.ma.masked_array(dice_scores, mask=np.equal(dice_scores, -1))
    mean_dice = np.mean(dice_score_ma, axis=0)
    print("DICE")
    print(mean_dice)
    # save_evaluation_results(["dice"], mean_dice, "dice_" + data_set_name + '.csv', res_path,
    #                         add_col=None, add_value=None)
    return save_evaluation_results(eval_df, ["dice"], mean_dice, "evaluation_performance_" + data_set_name + '.csv',
                                   res_path,
                                   add_col=None, add_value=None)


def compute_save_auc(eval_df, data_set_name, image_pred_method, res_path, image_labels, image_predictions, class_name):
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
    save_evaluation_results(eval_df, ['AUC_' + class_name], auc_all_classes_v1, 'evaluation_performance_' +
                            data_set_name + '.csv', res_path)

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


def compute_bag_prediction_as_training(has_bbox, predictions, patch_labels, pool_method, r):
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
                        compute_bag_prediction_lse_on_segmentation(nn_output=predictions, patch_labels=patch_labels,
                                                                   r=r),
                        compute_bag_prediction_lse(predictions, r))


def compute_bag_prediction(predictions, has_bbox, patch_labels, pool_method, r,
                           image_prediction_method='as_production'):
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


def do_transformation_masks_pascal(image_dir):
    """
    Transforms an image mask to size of 512x512. The resizing of the mask corresponds to the input size of the images to
     the NN network.
    :param image_dir: image path
    :return: returns resized image mask
    """
    img_width, img_height = load_img(image_dir, target_size=None, color_mode='rgb').size
    decrease_needed = image_larger_input(img_width, img_height, 512, 512)

    # IF one or both sides have bigger size than the input, then decrease is needed
    if decrease_needed:
        ratio = calculate_scale_ratio(img_width, img_height, 512, 512)
        assert ratio >= 1.00, "wrong ratio - it will increase image size"
        assert int(img_height / ratio) == 512 or int(img_width / ratio) == 512, \
            "error in computation"
        image = img_to_array(load_img(image_dir, target_size=(int(img_height / ratio), int(img_width / ratio)),
                                      color_mode='rgb'))
    else:
        # ELSE just open image in its original form
        image = img_to_array(load_img(image_dir, target_size=None, color_mode='rgb'))
    ### PADDING
    pad_needed = padding_needed(image)

    if pad_needed:
        image = pad_image(image, final_size_x=512, final_size_y=512)

    return image


def get_mask_img_ind(mask_path1, mask_path2, image_indices):
    """
    Gets a mask for a desired image index.
    Checking if a mask of the image is found in one two mask paths. If a mask is found, the masks
     is read and resized to the input image size.
    :param mask_path1: allowed mask path
    :param mask_path2: alternative mask path
    :param image_indices: the image index of the searched mask
    :return: Returns the image indeces of segmented images, together with their ground-truth masks, index of the image and
    parent directory.
    """
    masks = []
    images_ind = []
    mask_parent_path1 = mask_path1.split('/')[-1]
    mask_parent_path2 = mask_path2.split('/')[-1]
    indices = []
    parent_paths = []
    for img_ind in range(0, image_indices.shape[0]):
        parent_path = image_indices[img_ind].split('/')[-2]

        if parent_path == mask_parent_path1 or parent_path == mask_parent_path2:
            if parent_path == mask_parent_path1:
                masks_path = mask_path1
            else:
                masks_path = mask_path2
            try:
                image_mask = do_transformation_masks_pascal(
                    str(masks_path + "/" + image_indices[img_ind].split('/')[-1]))
                masks.append(image_mask)
                images_ind.append(image_indices[img_ind].split('/')[-1])
                indices.append(img_ind)
                parent_paths.append(parent_path)
            except:
                print("Image was not found: " + str(masks_path + "/" + image_indices[img_ind].split('/')[-1]))
    return masks, images_ind, indices, parent_paths


def transform_pixels_to_patches(binary_masked, patch_pixels):
    annotation = np.zeros((16, 16))
    for height in range(0, 16):
        for width in range(0, 16):
            max_pixel = binary_masked[height * patch_pixels:(height + 1) * patch_pixels,
                        width * patch_pixels:(width + 1) * patch_pixels].max()

            annotation[height, width] = max_pixel
    return annotation


def convert_mask_image_to_binary_matrix(mask_parent_folder, masks):
    patch_pixels = 32
    annotations_coll = []
    for ind in range(masks.shape[0]):
        if mask_parent_folder[ind].lower() == 'tugraz_cars':
            ## BLUE CHANNEL is larger than 0
            red_green_channel = np.add(masks[ind][:, :, 0], masks[ind][:, :, 1])
            background_masked = np.ma.masked_where(red_green_channel > 0, red_green_channel).mask


        elif mask_parent_folder[ind].lower() == 'ethz_sideviews_cars':
            ## BLUE CHANNEL is 0
            white_color = masks[ind][:, :, 0]
            background_masked = np.ma.masked_where(white_color > 0, white_color).mask
        annotation_image = transform_pixels_to_patches(background_masked, patch_pixels)
        annotations_coll.append(annotation_image)
    return np.expand_dims(annotations_coll, axis=3)


def get_dice_and_accuracy_pascal(inst_labels, inst_pred):
    binary_instance_labels = np.array(inst_pred >= 0.5, dtype=bool)
    dice = compute_dice(binary_instance_labels, inst_labels, th_binarization=0.5)
    acc = compute_accuracy_on_segmentation(binary_instance_labels, inst_labels, 0.5, 0.1)
    return dice, acc


def process_mask_images_pascal(pascal_image_path, classifiers, res_path, predictions):
    '''
    This functions measures the instance performance only on the Pascal dataset
    :param config: configurations
    :param classifiers: list of classifier names
    :return: evaluation of the stability score against the instance performance.
             instance performance is measured with the dice score between predictions and available segmentations.
             Saves .csv files for dice score across classifiers for each image and visualizations of stability
             against instance performance.
    '''
    pascal_dir = str(Path(pascal_image_path).parent).replace("\\", "/")
    masks_path1 = pascal_dir + "/GTMasks/ETHZ_sideviews_cars"

    masks_path_2 = pascal_dir + "/GTMasks/TUGraz_cars"
    img_ind = np.load(res_path + 'image_indices_' + classifiers + '.npy', allow_pickle=True)
    gt_masks, image_name_to_keep, indices_to_keep, parents_folder = get_mask_img_ind(masks_path1,
                                                                                     masks_path_2, img_ind)

    masks_labels_coll = np.asarray(gt_masks)
    annotations_coll = convert_mask_image_to_binary_matrix(parents_folder, masks_labels_coll)
    annotations_coll = np.array(annotations_coll)

    dice, accuracy_iou = get_dice_and_accuracy_pascal(annotations_coll, predictions[indices_to_keep])
    return gt_masks, image_name_to_keep, indices_to_keep, parents_folder, dice, accuracy_iou


def evaluate_instance_performance_pascal(pascal_img_path, file_name, res_path, has_bbox):
    """
    Evaluates the instance performance of pascal dataset. This dataset is different with respect to the others,
    as the training does not consider the instance labels. The training is only on bag label. That requires different
    approach to calculate instance performance because we have no instance labels. The instance labels are all 0s or 1s
     depending on the bag label.
    :param pascal_img_path: path to the pascal images
    :param file_name: unique name of the prediction files
    :param res_path: results path
    :param predictions: raw predictions
    :return: Returns updated list of images that have available segmentation, dice score and accuracy from IOU
    based on IOU threshold of 0.1
    """
    predictions, image_indices, patch_labels = get_index_label_prediction(file_name, res_path)

    gt_masks, image_name_to_keep, indices_to_keep, parents_folder, dice_scores, accurate_localizations = \
        process_mask_images_pascal(pascal_img_path, file_name, res_path, predictions)
    has_bbox[indices_to_keep] = True
    return has_bbox, accurate_localizations, dice_scores, indices_to_keep


def save_results_table(image_prediction_method, image_labels, image_predictions, class_name, predictions_unique_name,
                       predict_res_path, has_bbox, accurate_localizations, dice_scores):
    eval_df = pd.DataFrame()
    eval_df = compute_save_accuracy_results(eval_df, predictions_unique_name, predict_res_path, has_bbox,
                                            accurate_localizations)
    eval_df = compute_save_dice_results(eval_df, predictions_unique_name, predict_res_path, dice_scores)
    compute_save_auc(eval_df, predictions_unique_name, image_prediction_method, predict_res_path, image_labels,
                     image_predictions, class_name)
