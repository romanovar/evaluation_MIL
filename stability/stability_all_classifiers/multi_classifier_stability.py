import numpy as np
from pathlib import Path
from keras_preprocessing.image import load_img, img_to_array
from sklearn.metrics import roc_auc_score, average_precision_score

from cnn.keras_utils import calculate_scale_ratio, image_larger_input
from cnn.preprocessor.load_data_mura import pad_image, padding_needed
from stability.preprocessor.preprocessing import load_predictions, indices_segmentation_images, \
    filter_predictions_files_on_indices, indices_positive_images
from stability.stability_2classifiers.stability_scores import compute_binary_stability_scores, \
    compute_continuous_stability_scores
from stability.utils import get_image_index, save_additional_kappa_scores_forthreshold, save_mean_stability, \
    save_mean_stability_auc, save_mean_dice
from stability.visualizations.visualization_utils import visualize_single_image_1class_5classifiers, \
    visualize_correlation_heatmap, combine_correlation_heatmaps_next_to_each_other, \
    make_scatterplot_with_errorbar, make_scatterplot_with_errorbar_v2, \
    visualize_5_classifiers_mura, visualize_5_classifiers


def get_matrix_total_nans_stability_score(stab_index_collection, total_images_collection, normalize):
    nan_matrix = np.count_nonzero(np.isnan(np.array(stab_index_collection).
                                           reshape(5, 5, len(total_images_collection[0]))), axis=-1)
    if normalize:
        return nan_matrix / len(total_images_collection[0])
    else:
        return nan_matrix


def ensure_file_contain_same(files, file_nr):
    for nr in range(0, file_nr):
        assert (np.asarray(files[0]) == np.asarray(files[nr])).all(), "files provided are not the same"


def get_nonduplicate_scores(total_images, models_nr, stability_score_coll):
    pairwise_stability_all_images = []
    # total_images = inst_labels[0].shape[0]
    # FOR EACH IMAGE, THE PREDICTIONS OF EACH CLASSIFIERS ARE COMPARED WITH THE WHOLE BAG AND AUC IS COMPUTED
    for image_ind in range(0, total_images):
        image_stab_scores = []
        for classifier_ind in range(0, models_nr):
            stab_scores = stability_score_coll[classifier_ind, classifier_ind + 1:, image_ind]
            image_stab_scores = np.concatenate((image_stab_scores, stab_scores))
        pairwise_stability_all_images.append(image_stab_scores)
    # TOTAL_IMAGES x 10 combinations of stability
    stability_res = np.array(pairwise_stability_all_images)

    return stability_res


def get_instance_auc_stability_score_all_classifiers(inst_labels, inst_pred, stability_coll):
    inst_labels = np.array(inst_labels)
    ensure_file_contain_same(inst_labels, len(inst_labels))

    image_auc_collection_all_classifiers = []
    pairwise_stability_all_images = []
    total_images = inst_labels[0].shape[0]
    # instance auc
    # FOR EACH IMAGE, THE PREDICTIONS OF EACH CLASSIFIERS ARE COMPARED WITH THE WHOLE BAG AND AUC IS COMPUTED
    for image_ind in range(0, total_images):
        all_instances_labels = inst_labels[0].reshape(total_images, -1)
        image_auc_collection = []
        image_stab_scores = []

        for classifier_ind in range(0, 5):
            inst_predictions_classifier = inst_pred[classifier_ind].reshape(total_images, -1)
            # inst_auc_classifiers = roc_auc_score(all_instances_labels[image_ind],
            #                                      inst_predictions_classifier[image_ind])
            # image_auc_collection.append(inst_auc_classifiers)
            stab_scores = stability_coll[classifier_ind, classifier_ind + 1:, image_ind]
            image_stab_scores = np.concatenate((image_stab_scores, stab_scores))

        pairwise_stability_all_images.append(image_stab_scores)
        image_auc_collection_all_classifiers.append(image_auc_collection)

    # TOTAL_IMAGES x TOTAL_CLASSIFIERS
    auc_res = np.array(image_auc_collection_all_classifiers)
    # TOTAL_IMAGES x 10 combinations of stability
    stability_res = np.array(pairwise_stability_all_images)

    return auc_res, stability_res


def compute_ap(inst_labels, inst_pred):
    image_ap_collection_all_classifiers = []
    # pairwise_stability_all_images = []
    total_images = inst_labels[0].shape[0]
    # instance auc
    # FOR EACH IMAGE, THE PREDICTIONS OF EACH CLASSIFIERS ARE COMPARED WITH THE WHOLE BAG AND AUC IS COMPUTED
    for image_ind in range(0, total_images):
        all_instances_labels = inst_labels[0].reshape(total_images, -1)
        ap_collection = []

        for classifier_ind in range(0, 5):
            inst_predictions_classifier = inst_pred[classifier_ind].reshape(total_images, -1)
            ap_classifiers = average_precision_score(all_instances_labels[image_ind],
                                                     inst_predictions_classifier[image_ind])
            ap_collection.append(ap_classifiers)

        image_ap_collection_all_classifiers.append(ap_collection)

    # TOTAL_IMAGES x TOTAL_CLASSIFIERS
    ap_res = np.array(image_ap_collection_all_classifiers)
    return ap_res


def compute_stability_scores(raw_predictions_collection, bin_threshold=0.5):
    '''
    Computes the stability scores between models. For models considering binary predictions (0/1 predictions),
    a threshold of 0.5 is used for the binarization of the raw predictions
    :param raw_predictions_collection: a collection where each element is a list with the raw predictions of a models
    :param bin_threshold: a threshold used for the binarization of raw predictions to binary ones.
                            Binary predictions are needed for some of the  stability scores.
    :return: Computes all stability scores - positive Jaccard, Corrected positive Jaccard, positive Jaccard with
    heuristic correction, positive overlap, corrected positive overlap, corrected IOU
    '''

    pos_jacc, corr_pos_jacc, corr_pos_jacc_heur, pos_overlap, corr_pos_overlap, corr_iou = \
        compute_binary_stability_scores(bin_threshold, raw_predictions_collection)
    pearson_correlation, spearman_rank_correlation = compute_continuous_stability_scores(
        raw_predictions_collection)
    return pos_jacc, corr_pos_jacc, corr_pos_jacc_heur, pos_overlap, corr_pos_overlap, corr_iou, \
           pearson_correlation, spearman_rank_correlation


def generate_visualizations_stability(config, visualize_per_image,pos_jacc, corr_pos_jacc, corr_pos_jacc_heur,
                                      pos_overlap, corr_pos_overlap, corr_iou, pearson_correlation,
                                      spearman_rank_correlation, image_labels_collection,
                                      image_index_collection, raw_predictions_collection,
                                      samples_identifier):
    """
    Generates following visualizations:
                1. Optionally a visualization where predictions of each model are shown next to each other for each image.
                2. Heatmaps showing the number/relative quantity of NaN stability score between 2 models.
                    This may be interesting to compute, as NAN values are skipped from the aggregations (e.g. mean values)
                     of stability scores.
                3. Heatmaps showing the average stability for all images. The heatmaps are between any 2 pair of models
                and the average score of each stability index (possitive Jaccard, corrected positive Jaccard, etc) is on
                 a separate heatmap.
                 4. Lastly, computation of average stability score across all models per image.
                 This information is saved in a file, allowing further analysis. This information reveal differences in
                  values of the stability scores for the same image.


    :param config: configuration file
    :param visualize_per_image: If True: Predictions of each model are shown next to each other for each image.
    :param pos_jacc: a list of positive jaccard scores
    :param corr_pos_jacc: a list of corrected positive jaccard scores
    :param corr_pos_jacc_heur: a list of positive jaccard with heuristic correction
    :param pos_overlap: a list of positive overlap scores
    :param corr_pos_overlap: a list ofcorrected positive overlap scores
    :param corr_iou: a list of corrected iou scores
    :param pearson_correlation: a list of pearson correlation coefficient
    :param spearman_rank_correlation: a list of  spearman_rank_correlation coefficient
    :param image_labels_collection: a collection of the image labels per model.
    :param image_index_collection: a collection of image index per model
    :param raw_predictions_collection: colleciton of raw predictions per model
    :param samples_identifier: an identifier used when saving graphs. Used to denote which samples are used for the
                                analysis - all images, only positive images, only segmented images
    :return:
    """
    image_path = config['image_path']
    stability_res_path = config['stability_results']
    xray_dataset = config['use_xray_dataset']
    use_pascal_dataset = config['use_pascal_dataset']
    class_name = config['class_name']

    if xray_dataset:
        dataset_identifier = 'xray'
    elif use_pascal_dataset:
        dataset_identifier = 'pascal'
    else:
        dataset_identifier = 'mura'

    dataset_identifier += samples_identifier

    # Reshaping the stability scores by (# models compared, # models compared, # samples compared).
    # Taking an index of the array in the 3rd dimension results in a 2 dimensional array
    # with size of (#models compared, # model compared).
    # Each element of the 2D array keeps the stability score between 2 specific models.
    # Consider the following example of the array for specific image (indexing on the 3rd dimension)
    # E.g.  Mod1 Mod2 Mod3    The example assumes a comparison between the predictions of 3 different models.
    #  Mod1 | a  | b  | c |
    #  Mod2 | b  | d  | e |
    #  Mod3 | c  | e  | f |
    reshaped_jacc_coll = np.asarray(pos_jacc).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_jacc_coll = np.asarray(corr_pos_jacc).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_correlation).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_iou = np.asarray(corr_iou).reshape(5, 5, len(image_index_collection[0]))

    xyaxis = ['classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
    if visualize_per_image:
        visualize_5_classifiers(xray_dataset, use_pascal_dataset, image_index_collection, image_labels_collection,
                                raw_predictions_collection, image_path, stability_res_path, class_name, '_test_5_class')
    ## ADD inst AUC vs score
    ma_corr_jaccard_images = np.ma.masked_array(reshaped_corr_jacc_coll, np.isnan(reshaped_corr_jacc_coll))
    ma_jaccard_images = np.ma.masked_array(reshaped_jacc_coll, np.isnan(reshaped_jacc_coll))
    ma_corr_iou = np.ma.masked_array(reshaped_corr_iou, np.isnan(reshaped_corr_iou))
    ma_spearman = np.ma.masked_array(reshaped_spearman_coll, np.isnan(reshaped_spearman_coll))

    ############ visualizing NANs of corrected jaccard ###################
    nan_matrix_norm = get_matrix_total_nans_stability_score(corr_pos_jacc, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_norm, stability_res_path, '_corr_pos_jacc_nan_norm' + samples_identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix = get_matrix_total_nans_stability_score(corr_pos_jacc, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix, stability_res_path, '_corr_pos_jacc_nan' + samples_identifier, xyaxis,
                                  dropDuplicates=True)

    nan_matrix_jacc_norm = get_matrix_total_nans_stability_score(pos_jacc, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_jacc_norm, stability_res_path, '_pos_jacc_nan_norm' + samples_identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_jacc = get_matrix_total_nans_stability_score(pos_jacc, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix_jacc, stability_res_path, '_pos_jacc_nan' + samples_identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_spearman = get_matrix_total_nans_stability_score(spearman_rank_correlation,
                                                                image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix_spearman, stability_res_path, '_spearman_nan' + samples_identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_spearman_norm = get_matrix_total_nans_stability_score(spearman_rank_correlation,
                                                                     image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_spearman_norm, stability_res_path, '_spearman_nan_norm' + samples_identifier,
                                  xyaxis,
                                  dropDuplicates=True)

    ##### AVERAGE STABILITY ACROSS ALL IMAGES ##############
    average_corr_jacc_index_inst = np.average(ma_corr_jaccard_images, axis=-1)
    visualize_correlation_heatmap(average_corr_jacc_index_inst, stability_res_path,
                                  '_mean_stability_' + str("corr_jaccard") + '_' + samples_identifier,
                                  xyaxis, dropDuplicates=True)
    average_jacc_inst = np.average(ma_jaccard_images, axis=-1)
    visualize_correlation_heatmap(average_jacc_inst, stability_res_path,
                                  '_mean_stability_' + str("jaccard") + '_' + samples_identifier,
                                  xyaxis, dropDuplicates=True)

    average_iou_inst = np.average(ma_corr_iou, axis=-1)
    visualize_correlation_heatmap(average_iou_inst, stability_res_path,
                                  '_mean_stability_' + str("iou") + '_' + samples_identifier,
                                  xyaxis, dropDuplicates=True)

    save_additional_kappa_scores_forthreshold(0.5, raw_predictions_collection, image_index_collection,
                                              reshaped_corr_iou,
                                              reshaped_corr_jacc_coll, stability_res_path)

    avg_abs_sprearman = np.average(abs(ma_spearman), axis=-1)
    visualize_correlation_heatmap(avg_abs_sprearman, stability_res_path,
                                  '_mean_stability_' + str("abs_spearman") + '_' + samples_identifier,
                                  xyaxis, dropDuplicates=True)
    avg_spearman = np.average(ma_spearman, axis=-1)
    visualize_correlation_heatmap(avg_spearman, stability_res_path,
                                  '_mean_stability_' + str("spearman") + '_' + samples_identifier,
                                  xyaxis, dropDuplicates=True)

    #### AVERAGE ACROSS ALL CLASSIFIERS - PER IMAGE #######
    mask_repetition = np.ones((ma_corr_jaccard_images.shape), dtype=bool)
    for i in range(0, 4):
        for j in range(i + 1, 5):
            mask_repetition[i, j, :] = False
    mean_all_classifiers_corr_jacc = np.mean(np.ma.masked_array(ma_corr_jaccard_images,
                                                                mask=mask_repetition), axis=(0, 1))
    mean_all_classifiers_jacc = np.mean(np.ma.masked_array(ma_jaccard_images, mask=mask_repetition), axis=(0, 1))

    mean_all_classifiers_iou = np.mean(np.ma.masked_array(ma_corr_iou, mask=mask_repetition), axis=(0, 1))
    mean_all_classifiers_spearman = np.mean(np.ma.masked_array(ma_spearman, mask=mask_repetition), axis=(0, 1))

    save_mean_stability(image_index_collection[0], mean_all_classifiers_jacc, mean_all_classifiers_corr_jacc,
                        mean_all_classifiers_iou, mean_all_classifiers_spearman, stability_res_path, dataset_identifier)


def load_and_filter_predictions(config, classifiers, only_segmentation_images, only_positive_images):
    '''
    Loads prediction files and filters on specific samples, which are used to compute stability
    :param config: config file
    :param classifiers: list with all classifier names
    :param only_segmentation_images: True: analysis is done only on images with segmentation,
                                    False: analysis is done on all images
    :param only_positive_images: this parameter is considered only if only_segmentation_images is FALSE
                True: analysis done on image with positive label
                False: analysis is done on all images
    :return: Returns the suitable rows of the subset desired
    '''

    prediction_results_path = config['prediction_results_path']

    image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
    bag_predictions_collection, _ = load_predictions(classifiers, prediction_results_path)

    if only_segmentation_images:
        filtered_idx_collection = indices_segmentation_images(image_labels_collection)
        identifier = "_segmented_img"
        image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
        bag_predictions_collection = filter_predictions_files_on_indices(image_labels_collection, image_index_collection,
                                                                         raw_predictions_collection,
                                                                         bag_predictions_collection, bag_labels_collection,
                                                                         filtered_idx_collection)
        return image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
               bag_predictions_collection, identifier

    elif only_positive_images:
        filtered_idx_collection = indices_positive_images(bag_labels_collection)
        identifier = "_pos_img"
        image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
        bag_predictions_collection = filter_predictions_files_on_indices(image_labels_collection, image_index_collection,
                                                                         raw_predictions_collection,
                                                                         bag_predictions_collection, bag_labels_collection,
                                                                         filtered_idx_collection)
        return image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
               bag_predictions_collection, identifier
    else:
        identifier = "_all_img"
        return image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
               bag_predictions_collection, identifier


def generate_visualizations_instance_level(config,pos_jacc, corr_pos_jacc, corr_pos_jacc_heur,
                                      pos_overlap, corr_pos_overlap, corr_iou, pearson_correlation,
                                      spearman_rank_correlation, image_labels_collection,
                                      image_index_collection, raw_predictions_collection, dice_scores,
                                      stability_res_path):
    """
    Visualizing scatter plots with the dice score against some stability scores.
     In this way we can sees ome patterns between well
     segmented images (high avg dice score and low std dev of dice) and the stability score.
     Another interesting aspect is the behaviour of stability score for bag with high std dev of dice - or how stable
     are images for which models suggest various performance. Some visualization include std dev of the x-axis data,
     other on the y-axis. Some of the visualizations support line of best fit as well.
    :param config:
    :param classifiers: list of results names
    :param only_segmentation_images: True: only images with available segmentation to consider. This should be True
    :param only_positive_images: True: only images with positive label to consider
    :return: scatter plots between dice score and some stability scores.
    """


    reshaped_jacc_coll = np.asarray(pos_jacc).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_jacc_coll = np.asarray(corr_pos_jacc).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_correlation).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_iou = np.asarray(corr_iou).reshape(5, 5, len(image_index_collection[0]))

    nonduplicate_corr_pos_jacc = get_nonduplicate_scores(len(image_index_collection[0]), 5, reshaped_corr_jacc_coll)

    ap_res = compute_ap(image_labels_collection, raw_predictions_collection)

    avg_dice =np.mean(np.squeeze(dice_scores), axis=0)
    std_dev_dice = np.std(np.squeeze(dice_scores), axis=0)
    avg_stability_jacc = np.mean(np.ma.masked_array(nonduplicate_corr_pos_jacc, np.isnan(nonduplicate_corr_pos_jacc)), axis=1)
    stand_dev_stability_jacc = np.std(np.ma.masked_array(nonduplicate_corr_pos_jacc, np.isnan(nonduplicate_corr_pos_jacc)),
                                      axis=1)

    make_scatterplot_with_errorbar(avg_dice, 'mean dice', avg_stability_jacc, 'mean adjusted Positive0 Jaccard',
                                   stability_res_path, fitting_curve=False, y_errors=std_dev_dice,
                                   x_errors=stand_dev_stability_jacc,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_dice, 'mean dice', avg_stability_jacc, 'mean adjusted positive Jaccard',
                                   stability_res_path, fitting_curve=True, y_errors=std_dev_dice, x_errors=None,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_dice, 'mean dice', avg_stability_jacc, 'mean Aajusted Positive1 Jaccard',
                                   stability_res_path, fitting_curve=False, y_errors=None,
                                   x_errors=stand_dev_stability_jacc,
                                   error_bar=True, bin_threshold_prefix=0)

    nonduplicate_spear = get_nonduplicate_scores(len(image_index_collection[0]), 5, reshaped_spearman_coll)

    avg_stability_spear = np.mean(nonduplicate_spear, axis=1)
    stand_dev_stability_spear = np.std(nonduplicate_spear, axis=1)

    make_scatterplot_with_errorbar(avg_dice, 'mean dice', avg_stability_spear, 'mean Spearman0 correlation',
                                   stability_res_path, fitting_curve=False,
                                   y_errors=std_dev_dice, x_errors=stand_dev_stability_spear,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_dice, 'mean dice', avg_stability_spear, 'mean Spearman correlation',
                                   stability_res_path, fitting_curve=True, y_errors=std_dev_dice, x_errors=None,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_dice, 'mean dice', avg_stability_spear, 'mean Spearman1 correlation',
                                   stability_res_path, fitting_curve=False, y_errors=None,
                                   x_errors=stand_dev_stability_spear,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar_v2(avg_stability_jacc, avg_stability_spear, 'stability_score', avg_dice,
                                      "auc", stability_res_path, y_errors=stand_dev_stability_jacc,
                                      y_errors2=stand_dev_stability_spear, error_bar=False,
                                      bin_threshold_prefix=None, x_errors=None)

    mask_repetition = np.ones((reshaped_corr_jacc_coll.shape), dtype=bool)
    for i in range(0, 4):
        for j in range(i + 1, 5):
            mask_repetition[i, j, :] = False
    diag_classifiers_corr_jacc = np.ma.masked_array(reshaped_corr_jacc_coll, mask=mask_repetition)
    diag_masked_corr_jacc = np.ma.masked_array(diag_classifiers_corr_jacc, mask=np.isnan(diag_classifiers_corr_jacc))
    mean_corr_jacc = np.mean(diag_masked_corr_jacc, axis=(0, 1))

    diag_classifiers_spearman = np.ma.masked_array(reshaped_spearman_coll, mask=mask_repetition)
    diag_masked_spearman = np.ma.masked_array(diag_classifiers_spearman, mask=np.isnan(diag_classifiers_spearman))
    mean_spearman = np.mean(diag_masked_spearman, axis=(0, 1))
    assert mean_corr_jacc.all() == avg_stability_jacc.all(), "error"
    assert mean_spearman.all() == avg_stability_spear.all(), "error"
    #todo: delete inst auc functionality - do similar functionality for dice
    # save_mean_stability_auc(image_index_collection[0], auc_res, avg_stability_jacc,
    #                         avg_stability_spear, stability_res_path, 'bbox', ap_res)


def filter_segmentation_images_bbox_file(config, classifiers):
    prediction_results_path = config['prediction_results_path']

    image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
    bag_predictions_collection, bbox_collection = load_predictions(classifiers, prediction_results_path)

    for model_idx in range(1, len(bbox_collection)):
        assert (bbox_collection[model_idx] == bbox_collection[model_idx-1]).all(), "bbox files are not equal!"
    idx_to_filter_coll = []
    for model_idx in range(len(bbox_collection)):
        idx_to_filter_coll.append(np.where(bbox_collection[model_idx]==True)[0])

    return idx_to_filter_coll

# def stability_all_classifiers_instance_level_pascal(config, classifiers):
#     '''
#     This functions measures the instance performance only on the Pascal dataset
#     :param config: configurations
#     :param classifiers: list of classifier names
#     :return: evaluation of the stability score against the instance performance.
#              instance performance is measured with the dice score betweeen predictions and available segmentations.
#              Saves .csv files for dice score across classifiers for each image and visualizations of stability
#              against instance performance.
#     '''
#     stability_res_path = config['stability_results']
#     # pascal_image_path = config['pascal_image_path']
#     # pascal_dir = str(Path(pascal_image_path).parent).replace("\\", "/")
#     # masks_path1 = pascal_dir + "/GTMasks/ETHZ_sideviews_cars"
#     #
#     # masks_path_2 = pascal_dir + "/GTMasks/TUGraz_cars"
#     #
#     # prediction_results_path = config['prediction_results_path']
#     #
#     # masks_labels_coll = []
#     # image_name_coll = []
#     # indices_to_keep_coll = []
#     # for ind in range(5):
#     #     img_ind = np.load(prediction_results_path + 'image_indices_' + classifiers[ind], allow_pickle=True)
#     #     gt_masks, image_name_to_keep, indices_to_keep, parents_folder = filter_images_with_masks(masks_path1,
#     #                                                                                              masks_path_2, img_ind)
#     #     masks_labels_coll.append(gt_masks)
#     #     image_name_coll.append(image_name_to_keep)
#     #     indices_to_keep_coll.append(indices_to_keep)
#
#     # image_labels_collection, image_index_collection, raw_predictions_collection, \
#     # bag_labels_collection, bag_predictions_collection, identifier = get_analysis_data_masks_pascal(config, classifiers,
#     #                                                                                                indices_to_keep)
#
#     jacc_coll, corr_jacc_coll, _, _, _, corr_iou = compute_binary_stability_scores(0.5, raw_predictions_collection)
#
#     pearson_corr_collection, spearman_rank_corr_collection = compute_continuous_stability_scores(
#         raw_predictions_collection)
#
#     reshaped_jacc_coll = np.asarray(jacc_coll).reshape(5, 5, len(image_index_collection[0]))
#     reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
#     reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, len(image_index_collection[0]))
#     reshaped_corr_iou = np.asarray(corr_iou).reshape(5, 5, len(image_index_collection[0]))
#
#     # masks_labels_coll = np.asarray(masks_labels_coll)
#     # annotations_coll = convert_mask_image_to_binary_matrix(parents_folder, masks_labels_coll)
#     # annotations_coll = np.array(annotations_coll)
#     # auc_res, stability_res_corr_jacc = get_instance_auc_stability_score_all_classifiers(annotations_coll,
#     #                                                                                     raw_predictions_collection,
#     #                                                                                     reshaped_corr_jacc_coll)
#
#     # dice, accuracy_iou = get_dice_and_accuracy_pascal(annotations_coll, raw_predictions_collection)
#
#     ap_res = compute_ap(annotations_coll, raw_predictions_collection)
#
#     # avg_auc = np.mean(auc_res, axis=1)
#     # stand_dev_auc = np.std(auc_res, axis=1)
#
#     avg_stability_jacc = np.mean(np.ma.masked_array(stability_res_corr_jacc, np.isnan(stability_res_corr_jacc)), axis=1)
#     stand_dev_stability_jacc = np.std(np.ma.masked_array(stability_res_corr_jacc, np.isnan(stability_res_corr_jacc)),
#                                       axis=1)
#
#     make_scatterplot_with_errorbar(np.mean(dice, axis=1), 'mean DICE', avg_stability_jacc,
#                                    'DICE_mean adjusted Positive0 Jaccard',
#                                    stability_res_path, fitting_curve=False, y_errors=np.std(dice, axis=1),
#                                    x_errors=None, error_bar=True, bin_threshold_prefix=0)
#
#     make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_jacc, 'mean adjusted positive Jaccard',
#                                    stability_res_path, fitting_curve=False, y_errors=stand_dev_auc, x_errors=None,
#                                    error_bar=True, bin_threshold_prefix=0)
#
#     _, stability_res_spear = get_instance_auc_stability_score_all_classifiers(annotations_coll,
#                                                                               raw_predictions_collection,
#                                                                               reshaped_spearman_coll)
#
#     avg_stability_spear = np.mean(stability_res_spear, axis=1)
#     stand_dev_stability_spear = np.std(stability_res_spear, axis=1)
#
#     make_scatterplot_with_errorbar(np.mean(dice, axis=1), 'mean instance AUC', avg_stability_spear,
#                                    'dice_mean Spearman0 correlation',
#                                    stability_res_path, fitting_curve=False,
#                                    y_errors=stand_dev_auc, x_errors=stand_dev_stability_spear,
#                                    error_bar=True, bin_threshold_prefix=0)
#     make_scatterplot_with_errorbar(np.mean(dice, axis=1), 'mean DICE', avg_stability_spear, 'DICE_mean spear',
#                                    stability_res_path, fitting_curve=False, y_errors=np.std(dice, axis=1),
#                                    x_errors=None, error_bar=True, bin_threshold_prefix=0)
#
#     make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_spear, 'mean Spearman correlation',
#                                    stability_res_path, fitting_curve=False, y_errors=stand_dev_auc, x_errors=None,
#                                    error_bar=True, bin_threshold_prefix=0)
#     make_scatterplot_with_errorbar(np.mean(dice, axis=1), 'mean DICE', avg_stability_spear,
#                                    'DICE_mean_spearman',
#                                    stability_res_path, fitting_curve=True, y_errors=np.std(dice, axis=1),
#                                    x_errors=None, error_bar=True, bin_threshold_prefix=0)
#     mask_repetition = np.ones(reshaped_corr_jacc_coll.shape, dtype=bool)
#     for i in range(0, 4):
#         for j in range(i + 1, 5):
#             mask_repetition[i, j, :] = False
#     diag_classifiers_corr_jacc = np.ma.masked_array(reshaped_corr_jacc_coll, mask=mask_repetition)
#     diag_masked_corr_jacc = np.ma.masked_array(diag_classifiers_corr_jacc, mask=np.isnan(diag_classifiers_corr_jacc))
#     mean_corr_jacc = np.mean(diag_masked_corr_jacc, axis=(0, 1))
#
#     diag_classifiers_spearman = np.ma.masked_array(reshaped_spearman_coll, mask=mask_repetition)
#     diag_masked_spearman = np.ma.masked_array(diag_classifiers_spearman, mask=np.isnan(diag_classifiers_spearman))
#     mean_spearman = np.mean(diag_masked_spearman, axis=(0, 1))
#     assert mean_corr_jacc.all() == avg_stability_jacc.all(), "error"
#     assert mean_spearman.all() == avg_stability_spear.all(), "error"
#     save_mean_stability_auc(image_index_collection[0], auc_res, avg_stability_jacc,
#                             avg_stability_spear, stability_res_path, 'bbox', ap_res)
#
#     save_mean_dice(image_index_collection[0], dice, accuracy_iou, stability_res_path, 'bbox')
