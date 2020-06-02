import argparse

import yaml

from stability.preprocessor.preprocessing import load_filter_dice_scores, indices_segmentation_images, \
    filter_predictions_files_on_indices, load_and_filter_predictions, filter_segmentation_images_bbox_file
from stability.stability_2classifiers.stability_scores import compute_stability_scores
from stability.visualizations.visualization_utils import generate_visualizations_stability, \
    generate_visualizations_instance_level


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)
xray_dataset = config['use_xray_dataset']
use_pascal_dataset = config['use_pascal_dataset']
res_path = config['results_path']

set_name1 ='Cardiomegaly_test_set_CV0_0Cardiomegaly_0.95'
set_name2 = 'Cardiomegaly_test_set_CV0_1Cardiomegaly_0.95'
set_name3 ='Cardiomegaly_test_set_CV0_2Cardiomegaly_0.95'
set_name4 = 'Cardiomegaly_test_set_CV0_3Cardiomegaly_0.95'
set_name5 ='Cardiomegaly_test_set_CV0_4Cardiomegaly_0.95'


classifiers = [set_name1, set_name2, set_name3, set_name4, set_name5]

if xray_dataset:

    image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
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
                                      image_labels_collection=image_labels_collection,
                                      image_index_collection=image_index_collection,
                                      raw_predictions_collection=raw_predictions_collection,
                                      samples_identifier=identifier)

    ### Stability on instance level
    image_labels_segm_images, image_index_segm_images, raw_predictions_segm_images, bag_labels_segm_images, \
    bag_predictions_segm_images, identifier_segm_images = load_and_filter_predictions(config,
                                                                                      classifiers,
                                                                                      only_segmentation_images=True,
                                                                                      only_positive_images=False)
    filtered_idx_collection = indices_segmentation_images(image_labels_collection)

    dice_scores = load_filter_dice_scores(classifiers, filtered_idx_collection, res_path)

    pos_jacc_segm_img, corr_pos_jacc_segm_img, corr_pos_jacc_heur_segm_img, pos_overlap_segm_img, \
    corr_pos_overlap_segm_img, corr_iou_segm_img, pearson_correlation_segm_img, \
    spearman_rank_correlation_segm_img = compute_stability_scores(raw_predictions_segm_images)

    generate_visualizations_instance_level(config, pos_jacc_segm_img, corr_pos_jacc_segm_img, corr_pos_jacc_heur_segm_img,
                                           pos_overlap_segm_img, corr_pos_overlap_segm_img, corr_iou_segm_img,
                                           pearson_correlation_segm_img, spearman_rank_correlation_segm_img,
                                           image_labels_segm_images, image_index_segm_images, raw_predictions_segm_images,
                                           dice_scores, res_path)

elif use_pascal_dataset:
    image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
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
                                      image_labels_collection=image_labels_collection,
                                      image_index_collection=image_index_collection,
                                      raw_predictions_collection=raw_predictions_collection,
                                      samples_identifier=identifier)

    ### Stability on instance level
    filtered_idx_collection = filter_segmentation_images_bbox_file(config, classifiers)

    identifier = "_segmented_img"
    image_labels_segm_img, image_index_segm_img, raw_predictions_segm_img, bag_labels_segm_img, \
    bag_predictions_segm_img = filter_predictions_files_on_indices(image_labels_collection, image_index_collection,
                                                                   raw_predictions_collection,
                                                                   bag_predictions_collection, bag_labels_collection,
                                                                   filtered_idx_collection)
    dice_scores = load_filter_dice_scores(classifiers, filtered_idx_collection, res_path)

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
    image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
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
                                      image_labels_collection=image_labels_collection,
                                      image_index_collection=image_index_collection,
                                      raw_predictions_collection=raw_predictions_collection,
                                      samples_identifier=identifier)

