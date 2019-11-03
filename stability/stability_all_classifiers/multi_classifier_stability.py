import numpy as np

from stability.preprocessor.preprocessing import load_predictions_v2, indices_segmentation_images_v2, \
    filter_predictions_files_segmentation_images_v2
from stability.stability_2classifiers.stability_2classifiers import get_binary_scores_forthreshold_v2, \
    compute_correlation_scores_v2
from stability.visualizations.visualization_utils import visualize_single_image_1class_5classifiers


def stability_all_classifiers(config, classifiers):
    image_path = config['image_path']
    image_path = 'C:/Users/s161590/Documents/Project_li/bbox_images/'

    prediction_results_path = config['prediction_results_path']

    stability_res_path = config['stability_results']

    all_img_labels, all_img_image_ind, all_img_raw_predictions =  load_predictions_v2(classifiers, prediction_results_path)
    bbox_ind_collection = indices_segmentation_images_v2(all_img_labels)
    bbox_img_labels_coll, bbox_img_ind_coll, bbox_img_raw_predictions = \
        filter_predictions_files_segmentation_images_v2(all_img_labels, all_img_image_ind,
                                                            all_img_raw_predictions, bbox_ind_collection)

    jaccard_coll, corr_jacc_coll, corr_jacc_pigeonhole_coll, overlap_coll, corr_overlap_coll, corr_iou_coll = \
        get_binary_scores_forthreshold_v2(0.5, bbox_img_raw_predictions)

    pearson_corr_collection, spearman_rank_corr_collection = compute_correlation_scores_v2(bbox_img_raw_predictions)

    print(np.asarray(corr_jacc_coll)[:, 0])
    reshaped_corr_jacc_coll = np.asarray(corr_jacc_coll).reshape(5, 5, 28)
    reshaped_spearman_coll = np.asarray(spearman_rank_corr_collection).reshape(5, 5, 28)

    print(reshaped_corr_jacc_coll[:, :, 0])

    # for idx in range(0, len(bbox_img_ind_coll[0])):
    #     img_ind = bbox_img_ind_coll[0][idx][-16:-4]
    #     xyaxis = [ 'classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
    #     xyaxis_short = [ 'Cl.1', 'Cl. 2', 'Cl. 3', 'Cl. 4', 'Cl. 5']
    #
    #     # print("index "+ str(idx))
    #     # print(img_ind)
    #     # print(reshaped_corr_jacc_coll[:, :, idx])
    #     if not np.isnan(reshaped_corr_jacc_coll[:, :, idx]).all():
    #         isp.visualize_correlation_heatmap(reshaped_corr_jacc_coll[:, :, idx], isp.results_path,
    #                                           '_classifiers_jaccard_'+str(img_ind),
    #                                           xyaxis, dropDuplicates = True)
    #
    #         isp.combine_correlation_heatmaps_next_to_each_other(reshaped_corr_jacc_coll[:, :, idx], reshaped_spearman_coll[:, :, idx],
    #                                                             "Corrected Jaccard","Spearman rank",
    #                                                             xyaxis_short, isp.results_path, str( img_ind),
    #                                                             drop_duplicates=True)
    #     isp.visualize_correlation_heatmap(reshaped_spearman_coll[:, :, idx], isp.results_path,
    #                                           '_classifiers_spearman_'+str(img_ind),
    #                                           xyaxis, dropDuplicates = True)

    visualize_single_image_1class_5classifiers(bbox_img_ind_coll,bbox_img_labels_coll, bbox_img_raw_predictions, image_path,
                                               stability_res_path, 'Cardiomegaly', "_test_5clas")