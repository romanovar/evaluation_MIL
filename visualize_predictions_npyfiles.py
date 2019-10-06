import yaml
import argparse
import keras_utils
import os
import stability_predictions

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
stability_predictions.raw_predictions_1
stability_predictions.raw_predictions_95

print(stability_predictions.image_ind_1[-3:])
stability_predictions.image_ind_95

stability_predictions.labels_1
stability_predictions.labels_95

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

keras_utils.visualize_single_image_1class_2predictions(stability_predictions.image_ind_1[-1:],
                                                       stability_predictions.labels_1[-1:],
                                                       stability_predictions.raw_predictions_1[-1:], "1",
                                                       stability_predictions.auc_coll1[-1:],
                                                       stability_predictions.raw_predictions_95[-1:], "0.95",
                                                       stability_predictions.auc_coll95[-1:],
                                                       path, results_path, 'Cardiomegaly', "_combo2",
                                                       stability_predictions.jaccard_indices[-1:],
                                                       stability_predictions.corr_col[-1:])

