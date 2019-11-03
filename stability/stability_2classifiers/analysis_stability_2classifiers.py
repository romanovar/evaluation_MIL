from stability.preprocessor.preprocessing import load_predictions, indices_segmentation_images, \
    filter_predictions_files_segmentation_images
from stability.stability_2classifiers.stability_2classifiers import calculate_auc_save_in_df, get_suffix_models, \
    prepare_dataframes_with_results, get_binary_scores_forthreshold, calculate_spearman_rank_coefficient_v2, \
    compute_save_correlation_scores, compute_binary_scores_with_allthresholds
from stability.visualizations.visualization_utils import visualize_correlation_heatmap, \
    plot_change_stability_varying_threshold


def calculate_vizualize_save_stability(set_name1, set_name2, config):
    skip_processing = config['skip_processing_labels']
    image_path = config['image_path']
    classication_labels_path = config['classication_labels_path']
    localization_labels_path = config['localization_labels_path']
    results_path = config['results_path']
    generated_images_path = config['generated_images_path']
    processed_labels_path = config['processed_labels_path']
    prediction_results_path = config['prediction_results_path']
    train_mode = config['train_mode']
    test_single_image = config['test_single_image']
    trained_models_path = config['trained_models_path']
    stability_res_path = config['stability_results']

    suffix_1, suffix_2, split_suffix_1, split_suffix_2 = get_suffix_models(set_name1, set_name2)

    all_labels1, all_image_ind1, all_raw_predictions1, all_labels2, all_image_ind2,all_raw_predictions2 =\
        load_predictions(set_name1, set_name2, prediction_results_path)

    bbox_indices1, bbox_indices2 = indices_segmentation_images(all_labels1, all_labels2)

    labels_1, image_ind_1, raw_predictions_1, labels_2, image_ind_2, raw_predictions_2 = \
        filter_predictions_files_segmentation_images(all_labels1, all_image_ind1, all_raw_predictions1, bbox_indices1,
                                                 all_labels2, all_image_ind2,all_raw_predictions2, bbox_indices2)
    print(image_ind_1)
    df_stability, df_auc = prepare_dataframes_with_results(image_ind_1)


    df_auc, a1 = calculate_auc_save_in_df(raw_predictions_1, labels_1, df_auc, suffix_1)

    df_auc, a2 = calculate_auc_save_in_df(raw_predictions_2, labels_2, df_auc, suffix_2 )

    df_stability = compute_binary_scores_with_allthresholds(raw_predictions_1, raw_predictions_2, df_stability, a1, a2,
                                                         res_path=stability_res_path, scatterplots=True)


    pearson_corr, spearman_rank_corr = compute_save_correlation_scores(raw_predictions_1, raw_predictions_2,df_stability,
                                                                       suffix_1, suffix_2, split_suffix_1,
                                                                       res_path=stability_res_path, auc1=a1, auc2=a2,
                                                                       scatterplots=True)

    jacc, corr_jacc, jacc_pigeonhole, overlap, corr_overlap, corr_iou = get_binary_scores_forthreshold(0.5, raw_predictions_1,
                                                                                                       raw_predictions_2)
    correlation_between_scores = calculate_spearman_rank_coefficient_v2([jacc, corr_jacc, jacc_pigeonhole, corr_iou, overlap,
                                                                         corr_overlap, pearson_corr, spearman_rank_corr])
    all_labels = ['pos Jaccard', 'Corr positive Jacc', 'Corr Jacc using Pigeonhole', 'Corr IoU', 'Overlap', 'Corr overlap',
                  'Pearson', 'Spearman rank']
    bin_labels =['pos Jaccard', 'Corr positive Jacc', 'Corr Jacc using Pigeonhole', 'Corr IoU', 'Overlap', 'Corr overlap']
    correlation_between_scores2 = calculate_spearman_rank_coefficient_v2([jacc, corr_jacc, jacc_pigeonhole, corr_iou, overlap,
                                                                         corr_overlap])

    print(correlation_between_scores)
    print(correlation_between_scores2)

    visualize_correlation_heatmap(correlation_between_scores, results_path, "all",all_labels, dropDuplicates=True)
    visualize_correlation_heatmap(correlation_between_scores2, results_path, "binary",bin_labels, dropDuplicates=True)

    plot_change_stability_varying_threshold(raw_predictions_1, raw_predictions_2, results_path, image_ind_1)
