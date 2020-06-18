import argparse

import yaml

from cnn.keras_utils import set_dataset_flag, build_path_results, make_directory
from stability.preprocessing import load_filter_dice_scores, indices_segmentation_images, \
    filter_predictions_files_on_indices, load_and_filter_predictions, filter_segmentation_images_bbox_file
from stability.stability_scores import compute_stability_scores
from stability.visualization_utils import generate_visualizations_stability, \
    generate_visualizations_instance_level


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)
dataset_name = config['dataset_name']
res_path = config['results_path']
pooling_operator = config['pooling_operator']

set_name1 ='test_set_CV1_0'
set_name2 = 'test_set_CV1_1'
set_name3 ='test_set_CV1_2'
set_name4 = 'test_set_CV1_3'
set_name5 ='test_set_CV1_4'

use_xray, use_pascal = set_dataset_flag(dataset_name)
classifiers = [set_name1, set_name2, set_name3, set_name4, set_name5]
parent_folder_predictions = 'subsets'
predictions_path = build_path_results(res_path, dataset_name, pooling_operator,
                                             script_suffix=parent_folder_predictions,
                                             result_suffix='predictions')
performance_path = build_path_results(res_path, dataset_name, pooling_operator,
                                             script_suffix=parent_folder_predictions,
                                             result_suffix='performance')
stability_path = build_path_results(res_path, dataset_name, pooling_operator,
                                             script_suffix=parent_folder_predictions,
                                             result_suffix='stability')
make_directory(stability_path)

if use_xray:
    instance_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
    bag_predictions_collection, identifier = load_and_filter_predictions(classifiers,
                                                                         only_segmentation_images=False,
                                                                         only_positive_images=False,
                                                                         predictions_path=predictions_path)

    pos_jacc, corr_pos_jacc, corr_pos_jacc_heur, pos_overlap, corr_pos_overlap, corr_iou, \
    pearson_correlation, spearman_rank_correlation = compute_stability_scores(raw_predictions_collection)

    generate_visualizations_stability(config, visualize_per_image=False, pos_jacc=pos_jacc, corr_pos_jacc=corr_pos_jacc,
                                      corr_pos_jacc_heur=corr_pos_jacc_heur,
                                      pos_overlap=pos_overlap, corr_pos_overlap=corr_pos_overlap, corr_iou=corr_iou,
                                      pearson_correlation=pearson_correlation,
                                      spearman_rank_correlation=spearman_rank_correlation,
                                      image_labels_collection=instance_labels_collection,
                                      image_index_collection=image_index_collection,
                                      raw_predictions_collection=raw_predictions_collection,
                                      samples_identifier=identifier, stability_path=stability_path)

    ### Stability on instance level
    image_labels_segm_images, image_index_segm_images, raw_predictions_segm_images, bag_labels_segm_images, \
    bag_predictions_segm_images, identifier_segm_images = load_and_filter_predictions(classifiers,
                                                                                      only_segmentation_images=True,
                                                                                      only_positive_images=False,
                                                                                      predictions_path=predictions_path)
    filtered_idx_collection = indices_segmentation_images(instance_labels_collection)

    dice_scores = load_filter_dice_scores(classifiers, filtered_idx_collection, predictions_path)

    pos_jacc_segm_img, corr_pos_jacc_segm_img, corr_pos_jacc_heur_segm_img, pos_overlap_segm_img, \
    corr_pos_overlap_segm_img, corr_iou_segm_img, pearson_correlation_segm_img, \
    spearman_rank_correlation_segm_img = compute_stability_scores(raw_predictions_segm_images)

    generate_visualizations_instance_level(config, pos_jacc_segm_img, corr_pos_jacc_segm_img, corr_pos_jacc_heur_segm_img,
                                           pos_overlap_segm_img, corr_pos_overlap_segm_img, corr_iou_segm_img,
                                           pearson_correlation_segm_img, spearman_rank_correlation_segm_img,
                                           image_labels_segm_images, image_index_segm_images, raw_predictions_segm_images,
                                           dice_scores, res_path)

elif use_pascal:
    instance_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
    bag_predictions_collection, identifier = load_and_filter_predictions(config, classifiers,
                                                                         only_segmentation_images=False,
                                                                         only_positive_images=False)

    pos_jacc, corr_pos_jacc, corr_pos_jacc_heur, pos_overlap, corr_pos_overlap, corr_iou, \
    pearson_correlation, spearman_rank_correlation = compute_stability_scores(raw_predictions_collection)

    generate_visualizations_stability(config, visualize_per_image=False, pos_jacc=pos_jacc, corr_pos_jacc=corr_pos_jacc,
                                      corr_pos_jacc_heur=corr_pos_jacc_heur,
                                      pos_overlap=pos_overlap, corr_pos_overlap=corr_pos_overlap, corr_iou=corr_iou,
                                      pearson_correlation=pearson_correlation,
                                      spearman_rank_correlation=spearman_rank_correlation,
                                      image_labels_collection=instance_labels_collection,
                                      image_index_collection=image_index_collection,
                                      raw_predictions_collection=raw_predictions_collection,
                                      samples_identifier=identifier, stability_path=stability_path)

    ### Stability on instance level
    filtered_idx_collection = filter_segmentation_images_bbox_file(config, classifiers)

    identifier = "_segmented_img"
    image_labels_segm_img, image_index_segm_img, raw_predictions_segm_img, bag_labels_segm_img, \
    bag_predictions_segm_img = filter_predictions_files_on_indices(instance_labels_collection, image_index_collection,
                                                                   raw_predictions_collection,
                                                                   bag_predictions_collection, bag_labels_collection,
                                                                   filtered_idx_collection)
    dice_scores = load_filter_dice_scores(classifiers, filtered_idx_collection, predictions_path)

    pos_jacc_segm_img, corr_pos_jacc_segm_img, corr_pos_jacc_heur_segm_img, pos_overlap_segm_img, \
    corr_pos_overlap_segm_img, corr_iou_segm_img, pearson_correlation_segm_img, \
    spearman_rank_correlation_segm_img = compute_stability_scores(raw_predictions_segm_img)

    generate_visualizations_instance_level(config, pos_jacc_segm_img, corr_pos_jacc_segm_img,
                                           corr_pos_jacc_heur_segm_img,
                                           pos_overlap_segm_img, corr_pos_overlap_segm_img, corr_iou_segm_img,
                                           pearson_correlation_segm_img, spearman_rank_correlation_segm_img,
                                           image_labels_segm_img, image_index_segm_img,
                                           raw_predictions_segm_img, dice_scores, res_path)
else:
    instance_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
    bag_predictions_collection, identifier = load_and_filter_predictions(config, classifiers,
                                                                         only_segmentation_images=False,
                                                                         only_positive_images=False)

    pos_jacc, corr_pos_jacc, corr_pos_jacc_heur, pos_overlap, corr_pos_overlap, corr_iou, \
    pearson_correlation, spearman_rank_correlation = compute_stability_scores(raw_predictions_collection)

    generate_visualizations_stability(config, visualize_per_image=False, pos_jacc=pos_jacc, corr_pos_jacc=corr_pos_jacc,
                                      corr_pos_jacc_heur=corr_pos_jacc_heur,
                                      pos_overlap=pos_overlap, corr_pos_overlap=corr_pos_overlap, corr_iou=corr_iou,
                                      pearson_correlation=pearson_correlation,
                                      spearman_rank_correlation=spearman_rank_correlation,
                                      image_labels_collection=instance_labels_collection,
                                      image_index_collection=image_index_collection,
                                      raw_predictions_collection=raw_predictions_collection,
                                      samples_identifier=identifier, stability_path=stability_path)

