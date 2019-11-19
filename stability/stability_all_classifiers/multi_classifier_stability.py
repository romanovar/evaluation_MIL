import numpy as np
from sklearn.metrics import roc_auc_score

from stability.preprocessor.preprocessing import load_predictions_v2, indices_segmentation_images_v2, \
    filter_predictions_files_on_indeces, indices_positive_images
from stability.stability_2classifiers.stability_2classifiers import get_binary_scores_forthreshold_v2, \
    compute_correlation_scores_v2
from stability.visualizations.visualization_utils import visualize_single_image_1class_5classifiers, \
    visualize_correlation_heatmap, combine_correlation_heatmaps_next_to_each_other, visualize_scatter_bag_auc_stability, \
    make_scatterplot_with_errorbar


def stability_all_classifiers_bag_level(config, classifiers, only_segmentation_images, only_positive_images):
    image_path = config['image_path']
    image_path = 'C:/Users/s161590/Documents/Project_li/bbox_images/'

    prediction_results_path = config['prediction_results_path']
    stability_res_path = config['stability_results']
    # # image_labels_collection, image_index_collection, raw_predictions_collection
    #
    # all_img_labels, all_img_image_ind, all_img_raw_predictions, all_bag_labels, all_bag_predictions = \
    #     load_predictions_v2(classifiers, prediction_results_path)
    #
    # if only_segmentation_images:
    #     bbox_ind_collection = indices_segmentation_images_v2(all_img_labels)
    #     image_labels_collection, image_index_collection, raw_predictions_collection,  \
    #     bag_labels_collection, bag_predictions_collection =\
    #         filter_predictions_files_segmentation_images_v2(all_img_labels, all_img_image_ind,
    #                                                         all_img_raw_predictions,
    #                                                         all_bag_predictions,  all_bag_labels, bbox_ind_collection)
    #     identifier = "_bbox"
    # else:
    #     image_labels_collection, image_index_collection, raw_predictions_collection,\
    #     bag_labels_collection, bag_predictions_collection =\
    #         all_img_labels, all_img_image_ind, all_img_raw_predictions, all_bag_labels, all_bag_predictions
    #     identifier = "_all_img"
    image_labels_collection, image_index_collection, raw_predictions_collection, \
    bag_labels_collection, bag_predictions_collection, identifier = get_analysis_data_subset(config,
                                                                                             classifiers,
                                                                                             only_segmentation_images,
                                                                                             only_positive_images)

    _, corr_jacc_coll, _, _, _,_ = get_binary_scores_forthreshold_v2(0.5, raw_predictions_collection)

    pearson_corr_collection, spearman_rank_corr_collection = compute_correlation_scores_v2(raw_predictions_collection)

    reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, len(image_index_collection[0]))

    xyaxis = [ 'classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
    xyaxis_short = [ 'Cl.1', 'Cl. 2', 'Cl. 3', 'Cl. 4', 'Cl. 5']


    ma_corr_jaccard_images = np.ma.masked_array(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    print(ma_corr_jaccard_images)

    # visualize_correlation_heatmap(average_corr_jaccard_images, stability_res_path,
    #                               '_avg_jaccard_' + identifier,
    #                               xyaxis, dropDuplicates=True)
    ############ visualizing NANs of corrected jaccard ###################
    nan_matrix_norm = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_norm, stability_res_path,  '_jacc_nan_norm', xyaxis, dropDuplicates=True)
    nan_matrix = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix, stability_res_path,  '_jacc_nan',  xyaxis, dropDuplicates=True)

    average_jacc_index_inst = np.average(ma_corr_jaccard_images, axis=-1)
    avg_abs_sprearman = np.average(abs(reshaped_spearman_coll), axis=-1)


    ###################### BAG AUC vs STABILITY ###################################
    bag_auc_all_cl = compute_auc_classifiers(bag_predictions_collection, bag_labels_collection)

    visualize_bag_vs_stability(bag_auc_all_cl, average_jacc_index_inst, 'bag_AUC', '',
                               'corr_jaccard'+identifier, stability_res_path)
    visualize_bag_vs_stability(bag_auc_all_cl, avg_abs_sprearman, 'bag_AUC', '',
                               'spearman'+identifier, stability_res_path)


def get_matrix_total_nans_stability_score(stab_index_collection, total_images_collection,normalize):
    nan_matrix = np.count_nonzero(np.isnan(np.array(stab_index_collection).
                                           reshape(5, 5, len(total_images_collection[0]))), axis=-1)
    if normalize:
        return nan_matrix/len(total_images_collection[0])
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
            if ind!= ind2:
                observations = [bag_auc_list[ind], bag_auc_list[ind2]]
                mean_point = np.mean(observations, axis=0)
                stand_point = np.std(observations, axis=0)
                means.append(mean_point)
                stdevs.append(stand_point)
                stabilities.append(stability[ind, ind2])
    # visualize_scatter_bag_auc_stability(bag_auc_list, stability, y_axis_el1, y_axis_el2)
    make_scatterplot_with_errorbar(means, 'mean_' + y_axis_el1 + '_' + y_axis_el2 + '_error',
                                   stabilities, x_axis_title,
                                   res_path, y_errors=stdevs, error_bar=True, bin_threshold_prefix=None)


def compute_and_visualize_average_instance_stability(reshaped_stability, res_path, identifier, index_name, xy_axis_title):
    average_index_all_images = np.average(reshaped_stability, axis=-1)
    visualize_correlation_heatmap(average_index_all_images, res_path,
                                  '_avg_inst_'+str(index_name) + '_'+ identifier,
                                  xy_axis_title, dropDuplicates=True)


def plot_inst_performance_vs_stability_score_all_classifiers(inst_labels, inst_pred, ):
    # instance auc
    inst_auc = compute_auc_classifiers(inst_pred, inst_labels)
    stability_score
    scatterplot_AUC_stabscore_v2(y_axis_collection1, y_axis_title1, y_axis_title2, x_axis_collection, x_axis_title,
                                 res_path, threshold)

def stability_all_classifiers_instance_level(config, classifiers, only_segmentation_images,
                                             only_positive_images, visualize_per_image):
    image_path = config['image_path']
    image_path = 'C:/Users/s161590/Documents/Project_li/bbox_images/'
    stability_res_path = config['stability_results']

    image_labels_collection, image_index_collection, raw_predictions_collection, \
    bag_labels_collection, bag_predictions_collection, identifier = get_analysis_data_subset(config,
                                                                                             classifiers,
                                                                                             only_segmentation_images,
                                                                                             only_positive_images)

    jacc_coll, corr_jacc_coll, _, _, _, corr_iou = get_binary_scores_forthreshold_v2(0.5, raw_predictions_collection)

    pearson_corr_collection, spearman_rank_corr_collection = compute_correlation_scores_v2(raw_predictions_collection)

    print(np.asarray(corr_jacc_coll)[:, 0])
    reshaped_jacc_coll = np.asarray(jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, len(image_index_collection[0]))
    reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, len(image_index_collection[0]))
    reshaped_corr_iou = np.asarray(corr_iou).reshape(5,5, len(image_index_collection[0]))
    # reshaped_corr_jacc_coll = np.ma.masked_array(corr_jacc_coll).reshape(5, 5, 28)

    print(reshaped_corr_jacc_coll[:, :, 0])
    xyaxis = [ 'classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
    xyaxis_short = [ 'Cl.1', 'Cl. 2', 'Cl. 3', 'Cl. 4', 'Cl. 5']
    if visualize_per_image:
        for idx in range(0, len(image_index_collection[0])):
            img_ind = image_index_collection[0][idx][-16:-4]


            print("index "+ str(idx))
            print(img_ind)
            print(reshaped_corr_jacc_coll[:, :, idx])
            if not np.isnan(reshaped_corr_jacc_coll[:, :, idx]).all():
            # if not reshaped_corr_jacc_coll[:, :, idx].mask.all():
                visualize_correlation_heatmap(reshaped_corr_jacc_coll[:, :, idx], stability_res_path,
                                                  '_classifiers_jaccard_'+str(img_ind),
                                                  xyaxis, dropDuplicates = True)

                combine_correlation_heatmaps_next_to_each_other(reshaped_corr_jacc_coll[:, :, idx], reshaped_spearman_coll[:, :, idx],
                                                                    "Corrected Jaccard","Spearman rank",
                                                                    xyaxis_short, stability_res_path, str( img_ind),
                                                                    drop_duplicates=True)
            visualize_correlation_heatmap(reshaped_spearman_coll[:, :, idx], stability_res_path,
                                                  '_classifiers_spearman_'+str(img_ind),
                                                  xyaxis, dropDuplicates = True)


        visualize_single_image_1class_5classifiers(image_index_collection,image_labels_collection,raw_predictions_collection,
                                                   image_path, stability_res_path, 'Cardiomegaly', "_test_5clas")
    ## ADD inst AUC vs score


    ma_corr_jaccard_images = np.ma.masked_array(reshaped_corr_jacc_coll, np.isnan(reshaped_corr_jacc_coll))
    print(ma_corr_jaccard_images)
    ma_jaccard_images = np.ma.masked_array(reshaped_jacc_coll,  np.isnan(reshaped_jacc_coll))
    ma_corr_iou = np.ma.masked_array(reshaped_corr_iou, np.isnan(reshaped_corr_iou))

    # visualize_correlation_heatmap(average_corr_jaccard_images, stability_res_path,
    #                               '_avg_jaccard_' + identifier,
    #                               xyaxis, dropDuplicates=True)
    ############ visualizing NANs of corrected jaccard ###################
    nan_matrix_norm = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_norm, stability_res_path,  '_corr_jacc_nan_norm'+identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix = get_matrix_total_nans_stability_score(corr_jacc_coll, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix, stability_res_path,  '_corr_jacc_nan'+identifier,  xyaxis,
                                  dropDuplicates=True)

    nan_matrix_jacc_norm = get_matrix_total_nans_stability_score(jacc_coll, image_index_collection, normalize=True)
    visualize_correlation_heatmap(nan_matrix_jacc_norm, stability_res_path,  '_jacc_nan_norm'+identifier, xyaxis,
                                  dropDuplicates=True)
    nan_matrix_jacc = get_matrix_total_nans_stability_score(jacc_coll, image_index_collection, normalize=False)
    visualize_correlation_heatmap(nan_matrix_jacc, stability_res_path,  '_jacc_nan'+identifier,  xyaxis,
                                  dropDuplicates=True)
    ##### AVERAGE INSTANCE AUC VS STABILITY ##############
    average_corr_jacc_index_inst = np.average(ma_corr_jaccard_images, axis=-1)
    visualize_correlation_heatmap(average_corr_jacc_index_inst, stability_res_path,
                                  '_avg_inst_' + str("corr_jaccard") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)
    average_jacc_inst = np.average(ma_jaccard_images, axis=-1)
    visualize_correlation_heatmap(average_jacc_inst, stability_res_path,
                                  '_avg_inst_' + str("jaccard") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)

    average_iou_inst = np.average(ma_corr_iou, axis=-1)
    visualize_correlation_heatmap(average_iou_inst, stability_res_path,
                                  '_avg_inst_' + str("iou") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)
    # compute_and_visualize_average_instance_stability(ma_corr_jaccard_images, stability_res_path, "corr_jaccard",
    #                                                  identifier, xyaxis)
    avg_abs_sprearman = np.average(abs(reshaped_spearman_coll), axis=-1)
    # compute_and_visualize_average_instance_stability(abs(reshaped_spearman_coll), stability_res_path, "spearman",
    #                                                  identifier, xyaxis)
    visualize_correlation_heatmap(avg_abs_sprearman, stability_res_path,
                                  '_avg_inst_' + str("abs_spearman") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)
    avg_spearman = np.average(reshaped_spearman_coll, axis=-1)
    visualize_correlation_heatmap(avg_spearman, stability_res_path,
                                  '_avg_inst_' + str("spearman") + '_' + identifier,
                                  xyaxis, dropDuplicates=True)


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
        image_labels_collection, image_index_collection, raw_predictions_collection,  \
        bag_labels_collection, bag_predictions_collection =\
            filter_predictions_files_on_indeces(all_img_labels, all_img_image_ind,
                                                all_img_raw_predictions,
                                                all_bag_predictions,  all_bag_labels, bbox_ind_collection)
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
            image_labels_collection, image_index_collection, raw_predictions_collection,\
            bag_labels_collection, bag_predictions_collection =\
                all_img_labels, all_img_image_ind, all_img_raw_predictions, all_bag_labels, all_bag_predictions
            identifier = "_all_img"

    return image_labels_collection, image_index_collection, raw_predictions_collection, \
           bag_labels_collection, bag_predictions_collection, identifier
