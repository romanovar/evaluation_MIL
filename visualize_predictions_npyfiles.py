import yaml
import argparse
import keras_utils
import os
import instance_stability_predictions

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

skip_processing = config['skip_processing_labels']
image_path = config['image_path']
classication_labels_path = config['classication_labels_path']
localization_labels_path = config['localization_labels_path']
results_path =config['results_path']
generated_images_path = config['generated_images_path']
processed_labels_path = config['processed_labels_path']
prediction_results_path = config['prediction_results_path']
train_mode = config['train_mode']
test_single_image = config['test_single_image']
trained_models_path = config['trained_models_path']


################################ LOAD PREDICTION FILES ##################################
instance_stability_predictions.raw_predictions_1
instance_stability_predictions.raw_predictions_2

instance_stability_predictions.image_ind_2

instance_stability_predictions.labels_1
instance_stability_predictions.labels_2

path = 'C:/Users/s161590/Documents/Project_li/bbox_images/'
# keras_utils.visualize_single_image_1class(stability_predictions.image_ind_1[-3:],
#                                           stability_predictions.raw_predictions_1[-3:],
#                                           stability_predictions.labels_1[-3:], path, results_path,
#                                           'Cardiomegaly', "_1", stability_predictions.auc_coll1[-3:],
#                                           stability_predictions.jaccard_indices[-3:], stability_predictions.corr_col[-3:])
#
# keras_utils.visualize_single_image_1class(stability_predictions.image_ind_95[-3:],
#                                           stability_predictions.raw_predictions_95[-3:],
#                                           stability_predictions.labels_95[-3:], path, results_path,
#                                           'Cardiomegaly', "_95",  stability_predictions.auc_coll1[-3:],
#                                           stability_predictions.jaccard_indices[-3:], stability_predictions.corr_col[-3:])
#

jaccard_indices, corrected_pos_jacc, corrected_jacc_pigeonhole, overlap_coeff, corrected_overlap, corrected_iou = \
    instance_stability_predictions.get_binary_scores_forthreshold(0.5, instance_stability_predictions.raw_predictions_1,
                                                                  instance_stability_predictions.raw_predictions_2)

keras_utils.visualize_single_image_1class_2predictions(instance_stability_predictions.image_ind_1[14:15],
                                                       instance_stability_predictions.labels_1[14:15],
                                                       instance_stability_predictions.raw_predictions_1[14:15], "0",
                                                       instance_stability_predictions.a1[14:15],
                                                       instance_stability_predictions.raw_predictions_2[14:15], "1",
                                                       instance_stability_predictions.a2[14:15],
                                                       path, results_path, 'Cardiomegaly', "_combo2",
                                                       jaccard_indices[14:15],
                                                       corrected_pos_jacc[14:15],
                                                       corrected_jacc_pigeonhole[14:15],
                                                       corrected_iou[14:15],
                                                       overlap_coeff[14:15],
                                                       corrected_overlap[14:15],
                                                       instance_stability_predictions.pearson_corr[14:15],
                                                       instance_stability_predictions.spearman_rank_corr[14:15])

keras_utils.visualize_single_image_1class_2predictions(instance_stability_predictions.image_ind_1,
                                                       instance_stability_predictions.labels_1,
                                                       instance_stability_predictions.raw_predictions_1, "0",
                                                       instance_stability_predictions.a1,
                                                       instance_stability_predictions.raw_predictions_2, "1",
                                                       instance_stability_predictions.a2, path, results_path,
                                                       'Cardiomegaly', "_combo2",
                                                       jaccard_indices,
                                                       corrected_pos_jacc,
                                                       corrected_jacc_pigeonhole,
                                                       corrected_iou,
                                                       overlap_coeff,
                                                       corrected_overlap,
                                                       instance_stability_predictions.pearson_corr,
                                                       instance_stability_predictions.spearman_rank_corr)
