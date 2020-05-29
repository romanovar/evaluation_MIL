import yaml
import argparse

from stability.stability_all_classifiers.multi_classifier_stability import compute_stability_scores, \
    stability_all_classifiers_instance_level, stability_all_classifiers_instance_level_pascal, \
    load_and_filter_predictions, generate_visualizations_stability


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

set_name1 ='Cardiomegaly_test_set_CV0_0Cardiomegaly_0.95.npy'
set_name2 = 'Cardiomegaly_test_set_CV0_1Cardiomegaly_0.95.npy'
set_name3 ='Cardiomegaly_test_set_CV0_1Cardiomegaly_0.95.npy'
set_name4 = 'Cardiomegaly_test_set_CV0_3Cardiomegaly_0.95.npy'
set_name5 ='Cardiomegaly_test_set_CV0_4Cardiomegaly_0.95.npy'


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


    stability_all_classifiers_instance_level(config, classifiers, only_segmentation_images=True, only_positive_images=False)

elif use_pascal_dataset:
    stability_all_classifiers_instance_level_pascal(config, classifiers)
else:
    compute_stability_scores(config, classifiers, only_segmentation_images=False, only_positive_images=True,
                             visualize_per_image=False)

