import numpy as np
from sklearn.metrics import roc_auc_score

from stability.preprocessor.preprocessing import load_predictions_v2, indices_segmentation_images_v2, \
    filter_predictions_files_on_indeces, indices_positive_images
from stability.stability_2classifiers.stability_2classifiers import compute_correlation_scores_v2, \
    get_binary_scores_forthreshold_v2
from stability.utils import get_image_index, save_additional_kappa_scores_forthreshold, save_mean_stability
from stability.visualizations.visualization_utils import visualize_single_image_1class_5classifiers, \
    visualize_correlation_heatmap, combine_correlation_heatmaps_next_to_each_other, \
    make_scatterplot_with_errorbar, scatterplot_AUC_stabscore_v2, make_scatterplot_with_errorbar_v2, \
    visualize_5_classifiers_mura, visualize_5_classifiers


def stability_all_classifiers_bag_level(config, classifiers, only_segmentation_images, only_positive_images):
    image_path = config['image_path']
    image_path = 'C:/Users/s161590/Documents/Project_li/bbox_images/'

    prediction_results_path = config['prediction_results_path']
    stability_res_path = config['stability_results']

    image_labels_collection, image_index_collection, raw_predictions_collection, \
    bag_labels_collection, bag_predictions_collection, identifier = get_analysis_data_subset(config,
                                                                                             classifiers,
                                                                                             only_segmentation_images,
                                                                                             only_positive_images)

    _, corr_jacc_coll, _, _, _, _ = get_binary_scores_forthreshold_v2(0.5, raw_predictions_collection)

    pearson_corr_collection, spearman_rank_corr_collection = compute_correlation_scores_v2(raw_predictions_collection)

    reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, len(image_index_collection[0]))
    # 5 x 5 x 28

    stability_res_corr_jacc, stability_jacc_classifiers = get_value_unique_combinations(reshaped_corr_jacc_coll)

    ########## AVERAGE  ############################
    mean_corr_jacc = np.mean(np.ma.masked_array(stability_res_corr_jacc, np.isnan(stability_res_corr_jacc)))
    total_nan_values_jacc = np.count_nonzero(np.isnan(np.array(stability_res_corr_jacc)))
    norm_nan_values_jacc = total_nan_values_jacc / len(stability_res_corr_jacc)

    stability_res_spearman, stability_res_spearman_classifiers = get_value_unique_combinations(reshaped_spearman_coll)
    mean_spearman = np.mean(stability_res_spearman)

    bag_auc_all_cl = compute_auc_classifiers(bag_predictions_collection, bag_labels_collection)
    mean_bag_auc = np.mean(bag_auc_all_cl)
    # print("bag shape"+str(len(bag_auc_all_cl)))
    # print(mean_bag_auc)
    # print(mean_corr_jacc,mean_spearman)

    ################ average across CLASSIFIERS #############################
    mean_corr_jacc_classifiers = np.mean(np.ma.masked_array(stability_jacc_classifiers,
                                                            np.isnan(stability_jacc_classifiers)), axis=1)
    mean_spearman_classifiers = np.mean(np.ma.masked_array(stability_res_spearman_classifiers,
                                                           np.isnan(stability_res_spearman_classifiers)), axis=1)
    make_scatterplot_with_errorbar_v2(np.asarray(mean_corr_jacc_classifiers), np.asarray(mean_spearman_classifiers),
                                      'stability score',
                                      np.asarray(bag_auc_all_cl),
                                      'bag auc', stability_res_path, y_errors=None, y_errors2=None, error_bar=False,
                                      bin_threshold_prefix=0, x_errors=None)
    xyaxis = ['classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
    xyaxis_short = ['Cl.1', 'Cl. 2', 'Cl. 3', 'Cl. 4', 'Cl. 5']

    ma_corr_jaccard_images = np.ma.masked_array(reshaped_corr_jacc_coll)
    print(ma_corr_jaccard_images)

    ############ visualizing NANs of corrected jaccard ###################
    nan_matrix_norm = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_norm, stability_res_path, '_jacc_nan_norm', xyaxis, dropDuplicates=True)
    nan_matrix = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix, stability_res_path, '_jacc_nan', xyaxis, dropDuplicates=True)

    average_jacc_index_inst = np.average(ma_corr_jaccard_images, axis=-1)
    avg_abs_sprearman = np.average(abs(reshaped_spearman_coll), axis=-1)

    # ###################### BAG AUC vs STABILITY ###################################

    visualize_bag_vs_stability(bag_auc_all_cl, average_jacc_index_inst, 'bag_AUC', '',
                               'corr_jaccard' + identifier, stability_res_path)
    visualize_bag_vs_stability(bag_auc_all_cl, avg_abs_sprearman, 'bag_AUC', '',
                               'spearman' + identifier, stability_res_path)


def get_matrix_total_nans_stability_score(stab_index_collection, total_images_collection, normalize):
    nan_matrix = np.count_nonzero(np.isnan(np.array(stab_index_collection).
                                           reshape(5, 5, len(total_images_collection[0]))), axis=-1)
    if normalize:
        return nan_matrix / len(total_images_collection[0])
    else:
        return nan_matrix


def compute_auc_classifiers(all_bag_predictions, all_bag_labels):
    auc_scores = []
    for ind in range(0, len(all_bag_predictions)):
        auc = roc_auc_score(all_bag_labels[ind], all_bag_predictions[ind])
        auc_scores.append(auc)
    print("all auc scores")
    print(auc_scores)
    return auc_scores


def visualize_bag_vs_stability(bag_auc_list, stability, y_axis_el1, y_axis_el2, x_axis_title, res_path):
    means = []
    stdevs = []
    stabilities = []
    for ind in range(0, len(bag_auc_list)):
        for ind2 in range(0, len(bag_auc_list)):
            if ind != ind2:
                observations = [bag_auc_list[ind], bag_auc_list[ind2]]
                mean_point = np.mean(observations, axis=0)
                stand_point = np.std(observations, axis=0)
                means.append(mean_point)
                stdevs.append(stand_point)
                stabilities.append(stability[ind, ind2])
    # visualize_scatter_bag_auc_stability(bag_auc_list, stability, y_axis_el1, y_axis_el2)
    make_scatterplot_with_errorbar(means, 'mean_' + y_axis_el1 + '_' + y_axis_el2 + '_error',
                                   stabilities, x_axis_title,
                                   res_path, fitting_curve=False, y_errors=stdevs, error_bar=True,
                                   bin_threshold_prefix=None)


def compute_and_visualize_average_instance_stability(reshaped_stability, res_path, identifier, index_name,
                                                     xy_axis_title):
    average_index_all_images = np.average(reshaped_stability, axis=-1)
    visualize_correlation_heatmap(average_index_all_images, res_path,
                                  '_avg_inst_' + str(index_name) + '_' + identifier,
                                  xy_axis_title, dropDuplicates=True)


def ensure_file_contain_same(files, file_nr):
    for nr in range(0, file_nr):
        assert (files[0] == files[nr]).all(), "files provided are not the same"


def get_value_unique_combinations(stability_coll):
    pairwise_stability_all_images = []
    all_stab_scores = []
    classifiers_stab_score = []
    for classifier_ind in range(0, 5):
        stab_scores = stability_coll[classifier_ind, classifier_ind + 1:, :]
        # stab_score_per_classifier = stability_coll[classifier_ind, :, :]
        # mean_stab_score = np.mean(np.ma.masked_array(stab_scores, np.isnan(stab_scores), axis=-1))
        # print(mean_stab_score)
        all_stab_scores = np.concatenate((all_stab_scores, stab_scores.reshape(-1)), axis=0)
        stab_score_per_classifier = np.delete(stability_coll[classifier_ind], classifier_ind, axis=0)
        classifiers_stab_score.append(stab_score_per_classifier.reshape(-1))
        # image_stab_scores = np.concatenate((image_stab_scores, mean_stab_score))
        # image_stab_scores.append(stab_scores.flatten())
    return all_stab_scores, classifiers_stab_score


def get_instance_auc_stability_score_all_classifiers(inst_labels, inst_pred, stability_coll):
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
        # test = []
        for classifier_ind in range(0, 5):
            inst_predictions_classifier = inst_pred[classifier_ind].reshape(total_images, -1)

            inst_auc_classifiers = roc_auc_score(all_instances_labels[image_ind],
                                                 inst_predictions_classifier[image_ind])
            image_auc_collection.append(inst_auc_classifiers)
            stab_scores = stability_coll[classifier_ind, classifier_ind + 1:, image_ind]
            image_stab_scores = np.concatenate((image_stab_scores, stab_scores))
            # image_stab_scores.append(stab_scores)
        pairwise_stability_all_images.append(image_stab_scores)
        image_auc_collection_all_classifiers.append(image_auc_collection)

    # TOTAL_IMAGES x TOTAL_CLASSIFIERS
    auc_res = np.array(image_auc_collection_all_classifiers)
    # TOTAL_IMAGES x 10 combinations of stability
    stability_res = np.array(pairwise_stability_all_images)
    return auc_res, stability_res


def stability_all_classifiers(config, classifiers, only_segmentation_images,
                              only_positive_images, visualize_per_image):
    '''

    :param config:
    :param classifiers:
    :param only_segmentation_images:
    :param only_positive_images:
    :param visualize_per_image: do we wat
    :return:
    '''
    image_path = config['image_path']
    stability_res_path = config['stability_results']
    xray_dataset = config['use_xray_dataset']
    class_name = config['class_name']
    if xray_dataset:
        dataset_identifier = 'xray'
    else:
        dataset_identifier = 'mura'

    image_labels_collection, image_index_collection, raw_predictions_collection, \
    bag_labels_collection, bag_predictions_collection, identifier = get_analysis_data_subset(config,
                                                                                             classifiers,
                                                                                             only_segmentation_images,
                                                                                             only_positive_images)
    print(image_labels_collection[0])
    dataset_identifier += identifier
    jacc_coll, corr_jacc_coll, _, _, _, corr_iou = get_binary_scores_forthreshold_v2(0.5, raw_predictions_collection)

    pearson_corr_collection, spearman_rank_corr_collection = compute_correlation_scores_v2(raw_predictions_collection)

    reshaped_jacc_coll = np.asarray(jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_iou = np.asarray(corr_iou).reshape(5, 5, len(image_index_collection[0]))
    # reshaped_corr_jacc_coll = np.ma.masked_array(corr_jacc_coll).reshape(5, 5, 28)

    xyaxis = ['classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
    xyaxis_short = ['Cl.1', 'Cl. 2', 'Cl. 3', 'Cl. 4', 'Cl. 5']
    if visualize_per_image:
        for idx in range(0, 2):  # len(image_index_collection[0])):
            img_ind = get_image_index(xray_dataset, image_index_collection[0], idx)

            print("index " + str(idx))
            print(img_ind)
            print(reshaped_corr_jacc_coll[:, :, idx])
            if not np.isnan(reshaped_corr_jacc_coll[:, :, idx]).all():
                visualize_correlation_heatmap(reshaped_corr_jacc_coll[:, :, idx], stability_res_path,
                                              '_classifiers_jaccard_' + str(img_ind),
                                              xyaxis, dropDuplicates=True)

                combine_correlation_heatmaps_next_to_each_other(reshaped_corr_jacc_coll[:, :, idx],
                                                                reshaped_spearman_coll[:, :, idx],
                                                                "Corrected Jaccard", "Spearman rank",
                                                                xyaxis_short, stability_res_path, str(img_ind),
                                                                drop_duplicates=True)
            visualize_correlation_heatmap(reshaped_spearman_coll[:, :, idx], stability_res_path,
                                          '_classifiers_spearman_' + str(img_ind),
                                          xyaxis, dropDuplicates=True)
        visualize_5_classifiers(xray_dataset, image_index_collection, image_labels_collection,
                                raw_predictions_collection, image_path, stability_res_path, class_name, '_test_5_class')
    ## ADD inst AUC vs score
    ma_corr_jaccard_images = np.ma.masked_array(reshaped_corr_jacc_coll, np.isnan(reshaped_corr_jacc_coll))
    print(ma_corr_jaccard_images)
    ma_jaccard_images = np.ma.masked_array(reshaped_jacc_coll, np.isnan(reshaped_jacc_coll))
    ma_corr_iou = np.ma.masked_array(reshaped_corr_iou, np.isnan(reshaped_corr_iou))
    ma_spearman = np.ma.masked_array(reshaped_spearman_coll, np.isnan(reshaped_spearman_coll))

    # visualize_correlation_heatmap(average_corr_jaccard_images, stability_res_path,
    #                               '_avg_jaccard_' + identifier,
    #                               xyaxis, dropDuplicates=True)
    ############ visualizing NANs of corrected jaccard ###################
    nan_matrix_norm = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_norm, stability_res_path, '_corr_jacc_nan_norm' + identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix, stability_res_path, '_corr_jacc_nan' + identifier, xyaxis,
                                  dropDuplicates=True)

    nan_matrix_jacc_norm = get_matrix_total_nans_stability_score(jacc_coll, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_jacc_norm, stability_res_path, '_jacc_nan_norm' + identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_jacc = get_matrix_total_nans_stability_score(jacc_coll, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix_jacc, stability_res_path, '_jacc_nan' + identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_spearman = get_matrix_total_nans_stability_score(spearman_rank_corr_collection,
                                                                image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix_spearman, stability_res_path, '_spearman_nan' + identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_spearman_norm = get_matrix_total_nans_stability_score(spearman_rank_corr_collection,
                                                                     image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_spearman_norm, stability_res_path, '_spearman_nan_norm' + identifier,
                                  xyaxis,
                                  dropDuplicates=True)

    ##### AVERAGE STABILITY ACROSS ALL IMAGES ##############
    average_corr_jacc_index_inst = np.average(ma_corr_jaccard_images, axis=-1)
    visualize_correlation_heatmap(average_corr_jacc_index_inst, stability_res_path,
                                  '_mean_stability_' + str("corr_jaccard") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)
    average_jacc_inst = np.average(ma_jaccard_images, axis=-1)
    visualize_correlation_heatmap(average_jacc_inst, stability_res_path,
                                  '_mean_stability_' + str("jaccard") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)

    average_iou_inst = np.average(ma_corr_iou, axis=-1)
    visualize_correlation_heatmap(average_iou_inst, stability_res_path,
                                  '_mean_stability_' + str("iou") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)

    save_additional_kappa_scores_forthreshold(0.5, raw_predictions_collection, image_index_collection,
                                              reshaped_corr_iou,
                                              reshaped_corr_jacc_coll, stability_res_path)




    # compute_and_visualize_average_instance_stability(ma_corr_jaccard_images, stability_res_path, "corr_jaccard",
    #                                                  identifier, xyaxis)
    avg_abs_sprearman = np.average(abs(ma_spearman), axis=-1)
    # compute_and_visualize_average_instance_stability(abs(reshaped_spearman_coll), stability_res_path, "spearman",
    #                                                  identifier, xyaxis)
    visualize_correlation_heatmap(avg_abs_sprearman, stability_res_path,
                                  '_mean_stability_' + str("abs_spearman") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)
    avg_spearman = np.average(ma_spearman, axis=-1)
    visualize_correlation_heatmap(avg_spearman, stability_res_path,
                                  '_mean_stability_' + str("spearman") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)


    #### AVERAGE ACROSS ALL CLASSIFIERS - PER IMAGE #######
    mean_all_classifiers_corr_jacc = np.average(np.reshape(ma_corr_jaccard_images, (5*5, -1)), axis=0)
    mean_all_classifiers_jacc = np.average(np.reshape(ma_jaccard_images, (5*5, -1)), axis=0)
    mean_all_classifiers_iou = np.average(np.reshape(ma_corr_iou, (5*5, -1)), axis=0)
    mean_all_classifiers_spearman = np.average(np.reshape(ma_spearman, (5*5, -1)), axis=0)

    save_mean_stability(image_index_collection[0], mean_all_classifiers_jacc, mean_all_classifiers_corr_jacc,
                        mean_all_classifiers_iou,mean_all_classifiers_spearman, stability_res_path, dataset_identifier)


def get_analysis_data_subset(config, classifiers, only_segmentation_images, only_positive_images):
    '''

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

    all_img_labels, all_img_image_ind, all_img_raw_predictions, all_bag_labels, all_bag_predictions = \
        load_predictions_v2(classifiers, prediction_results_path)

    if only_segmentation_images:
        bbox_ind_collection = indices_segmentation_images_v2(all_img_labels)
        image_labels_collection, image_index_collection, raw_predictions_collection, \
        bag_labels_collection, bag_predictions_collection = \
            filter_predictions_files_on_indeces(all_img_labels, all_img_image_ind,
                                                all_img_raw_predictions,
                                                all_bag_predictions, all_bag_labels, bbox_ind_collection)
        identifier = "_bbox"
    else:
        if only_positive_images:
            positive_ind_collection = indices_positive_images(all_bag_labels)
            image_labels_collection, image_index_collection, raw_predictions_collection, \
            bag_labels_collection, bag_predictions_collection = \
                filter_predictions_files_on_indeces(all_img_labels, all_img_image_ind,
                                                    all_img_raw_predictions,
                                                    all_bag_predictions, all_bag_labels, positive_ind_collection)
            identifier = "_pos_img"
            print("finished")
        else:
            image_labels_collection, image_index_collection, raw_predictions_collection, \
            bag_labels_collection, bag_predictions_collection = \
                all_img_labels, all_img_image_ind, all_img_raw_predictions, all_bag_labels, all_bag_predictions
            identifier = "_all_img"

    return image_labels_collection, image_index_collection, raw_predictions_collection, \
           bag_labels_collection, bag_predictions_collection, identifier


def stability_all_classifiers_instance_level(config, classifiers, only_segmentation_images, only_positive_images):
    image_path = config['image_path']
    # image_path = 'C:/Users/s161590/Documents/Project_li/bbox_images/'
    stability_res_path = config['stability_results']

    image_labels_collection, image_index_collection, raw_predictions_collection, \
    bag_labels_collection, bag_predictions_collection, identifier = get_analysis_data_subset(config,
                                                                                             classifiers,
                                                                                             only_segmentation_images,
                                                                                             only_positive_images)

    jacc_coll, corr_jacc_coll, _, _, _, corr_iou = get_binary_scores_forthreshold_v2(0.5, raw_predictions_collection)

    pearson_corr_collection, spearman_rank_corr_collection = compute_correlation_scores_v2(raw_predictions_collection)

    reshaped_jacc_coll = np.asarray(jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_iou = np.asarray(corr_iou).reshape(5, 5, len(image_index_collection[0]))
    # reshaped_corr_jacc_coll = np.ma.masked_array(corr_jacc_coll).reshape(5, 5, 28)
    save_additional_kappa_scores_forthreshold(0.5, raw_predictions_collection, image_index_collection,
                                              reshaped_corr_iou,
                                              reshaped_corr_jacc_coll, stability_res_path)
    auc_res, stability_res_corr_jacc = get_instance_auc_stability_score_all_classifiers(image_labels_collection,
                                                                                        raw_predictions_collection,
                                                                                        reshaped_corr_jacc_coll)
    avg_auc = np.mean(auc_res, axis=1)
    stand_dev_auc = np.std(auc_res, axis=1)

    avg_stability_jacc = np.mean(np.ma.masked_array(stability_res_corr_jacc, np.isnan(stability_res_corr_jacc)), axis=1)
    stand_dev_stability_jacc = np.std(np.ma.masked_array(stability_res_corr_jacc, np.isnan(stability_res_corr_jacc)),
                                      axis=1)

    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_jacc, 'mean adjusted Positive0 Jaccard',
                                   stability_res_path, fitting_curve=False, y_errors=stand_dev_auc,
                                   x_errors=stand_dev_stability_jacc,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_jacc, 'mean adjusted positive Jaccard',
                                   stability_res_path, fitting_curve=True, y_errors=stand_dev_auc, x_errors=None,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_jacc, 'mean Aajusted Positive1 Jaccard',
                                   stability_res_path, fitting_curve=False, y_errors=None, x_errors=stand_dev_stability_jacc,
                                   error_bar=True, bin_threshold_prefix=0)

    _, stability_res_spear = get_instance_auc_stability_score_all_classifiers(image_labels_collection,
                                                                              raw_predictions_collection,
                                                                              reshaped_spearman_coll)

    avg_stability_spear = np.mean(stability_res_spear, axis=1)
    stand_dev_stability_spear = np.std(stability_res_spear, axis=1)

    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_spear, 'mean Spearman0 correlation',
                                   stability_res_path, fitting_curve=False,
                                   y_errors=stand_dev_auc, x_errors=stand_dev_stability_spear,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_spear, 'mean Spearman correlation',
                                   stability_res_path, fitting_curve=True, y_errors=stand_dev_auc, x_errors=None,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar(avg_auc, 'mean instance AUC', avg_stability_spear, 'mean Spearman1 correlation',
                                   stability_res_path, fitting_curve=False, y_errors=None, x_errors=stand_dev_stability_spear,
                                   error_bar=True, bin_threshold_prefix=0)
    make_scatterplot_with_errorbar_v2(avg_stability_jacc, avg_stability_spear, 'stability_score', avg_auc,
                                      "auc", stability_res_path, y_errors=stand_dev_stability_jacc,
                                      y_errors2=stand_dev_stability_spear, error_bar=False,
                                      bin_threshold_prefix=None, x_errors=None)
