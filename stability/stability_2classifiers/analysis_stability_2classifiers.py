from stability.preprocessor.preprocessing import load_predictions, indices_segmentation_images, \
    filter_predictions_files_segmentation_images, indices_positive_images
from stability.stability_2classifiers.scores_2classifiers import calculate_kendallstau_coefficient_batch, \
    calculate_spearman_rank_coefficient
from stability.stability_2classifiers.stability_2classifiers import calculate_auc_save_in_df, get_suffix_models, \
    prepare_dataframes_with_results, get_binary_scores_forthreshold, calculate_spearman_rank_coefficient_v2, \
    compute_save_correlation_scores, compute_binary_scores_with_allthresholds
from stability.visualizations.visualization_utils import visualize_correlation_heatmap, \
    plot_change_stability_varying_threshold


def calculate_vizualize_save_stability(set_name1, set_name2, config):
    prediction_results_path = config['prediction_results_path']
    stability_res_path = config['stability_results']

    suffix_1, suffix_2, split_suffix_1, split_suffix_2 = get_suffix_models(set_name1, set_name2)

    all_labels1, all_image_ind1, all_raw_predictions1, all_bag_labels_1, all_bag_predictions_1, \
    all_labels2, all_image_ind2, all_raw_predictions2, all_bag_labels_2, all_bag_predictions_2= \
        load_predictions(set_name1, set_name2, prediction_results_path)

    bbox_indices1, bbox_indices2 = indices_segmentation_images(all_labels1, all_labels2)

    labels_1, image_ind_1, raw_predictions_1, labels_2, image_ind_2, raw_predictions_2 = \
        filter_predictions_files_segmentation_images(all_labels1, all_image_ind1, all_raw_predictions1, bbox_indices1,
                                                     all_labels2, all_image_ind2, all_raw_predictions2, bbox_indices2)
    print("image index ")
    print(image_ind_1[18])
    df_stability, df_auc = prepare_dataframes_with_results(image_ind_1)

    df_auc, a1 = calculate_auc_save_in_df(raw_predictions_1, labels_1, df_auc, suffix_1)

    df_auc, a2 = calculate_auc_save_in_df(raw_predictions_2, labels_2, df_auc, suffix_2)


    df_stability = compute_binary_scores_with_allthresholds(raw_predictions_1, raw_predictions_2, df_stability, a1, a2,
                                                            res_path=stability_res_path, scatterplots=True)

    pearson_corr, spearman_rank_corr, tau_coll = compute_save_correlation_scores(raw_predictions_1, raw_predictions_2,
                                                                                 df_stability,
                                                                                 suffix_1, suffix_2, split_suffix_1,
                                                                                 res_path=stability_res_path, auc1=a1,
                                                                                 auc2=a2,
                                                                                 scatterplots=True)
    # print("TAU")
    # idx = [[2, 3, 10, 18, 13, 20, 24, 26]]
    # test_idx = [0, 1, 2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14 ,15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27]

    # scatterplot_AUC_stabscore(np.array(a1)[test_idx], 'AUC1', np.array(a2)[test_idx], "AUC2",
    #                           np.array(tau_coll)[test_idx],
    #                           "test", stability_res_path, threshold=0)

    jacc, corr_jacc, jacc_pigeonhole, overlap, corr_overlap, corr_iou = \
        get_binary_scores_forthreshold(0.5, raw_predictions_1, raw_predictions_2)
    correlation_between_scores = calculate_spearman_rank_coefficient_v2([jacc, corr_jacc, jacc_pigeonhole,
                                                                         corr_iou, overlap,
                                                                         corr_overlap, pearson_corr, spearman_rank_corr,
                                                                         tau_coll])
    all_labels = ['pos Jaccard', 'Corr positive Jacc', 'Corr Jacc using Pigeonhole',
                  'Corr IoU', 'Overlap', 'Corr overlap', 'Pearson', 'Spearman rank', "kendall tau" ]

    bin_labels = ['pos Jaccard', 'Corr positive Jacc', 'Corr Jacc using Pigeonhole',
                  'Corr IoU', 'Overlap', 'Corr overlap']
    correlation_between_scores2 = calculate_spearman_rank_coefficient_v2([jacc, corr_jacc, jacc_pigeonhole,
                                                                          corr_iou, overlap, corr_overlap])

    print(correlation_between_scores)
    print(correlation_between_scores2)

    visualize_correlation_heatmap(correlation_between_scores, stability_res_path, "all", all_labels,
                                  dropDuplicates=True)
    visualize_correlation_heatmap(correlation_between_scores2, stability_res_path, "binary", bin_labels,
                                  dropDuplicates=True)

    plot_change_stability_varying_threshold(raw_predictions_1, raw_predictions_2, stability_res_path, image_ind_1)


def examine_correlation(set_name1, set_name2, config):
    prediction_results_path = config['prediction_results_path']
    stability_res_path = config['stability_results']

    suffix_1, suffix_2, split_suffix_1, split_suffix_2 = get_suffix_models(set_name1, set_name2)

    all_labels1, all_image_ind1, all_raw_predictions1, all_bag_labels1, all_bag_predictions1,\
    all_labels2, all_image_ind2, all_raw_predictions2,  all_bag_labels2, all_bag_predictions2= \
        load_predictions(set_name1, set_name2, prediction_results_path)
    identifier = "_all_pos"
    pos_ind_col = indices_positive_images([all_bag_labels1, all_bag_labels2])

    labels_1, image_ind_1, raw_predictions_1, labels_2, image_ind_2, raw_predictions_2 = \
        filter_predictions_files_segmentation_images(all_labels1, all_image_ind1, all_raw_predictions1,
                                                     pos_ind_col[0],
                                                     all_labels2, all_image_ind2, all_raw_predictions2,
                                                     pos_ind_col[1])
    tau_coll= calculate_kendallstau_coefficient_batch(raw_predictions_1, raw_predictions_2)
    spearman_corr_col = calculate_spearman_rank_coefficient(raw_predictions_1, raw_predictions_2)
    correlation_between_scores = calculate_spearman_rank_coefficient_v2([spearman_corr_col, tau_coll])
    print(correlation_between_scores)
