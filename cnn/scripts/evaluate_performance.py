import argparse
import yaml
from cnn import keras_preds
from cnn.keras_utils import build_path_results, make_directory


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

results_path = config['results_path']
class_name = config['class_name']
dataset_name = config['dataset_name']
pooling_operator = config['pooling_operator']


image_prediction_method = 'as_production'
predictions_unique_name = 'test_set_CV1_0'
predictions_folder_name = 'subsets' #this could be CV or exploratory_exp, all 3 of them can have prediction files

predictions_path = build_path_results(results_path, dataset_name, pooling_operator,
                                             script_suffix=predictions_folder_name,
                                             result_suffix='predictions')
performance_path = build_path_results(results_path, dataset_name, pooling_operator,
                                             script_suffix=predictions_folder_name,
                                             result_suffix='performance')
make_directory(performance_path)

pool_dict = {'nor': "nor",
             "lse": "lse",
             "lse01": "lse",
             "max": "max"
             }
r = {'nor': 0,
    "lse": 1.0,
    "lse01": 0.1,
     "max": 0
     }

image_labels, image_predictions, \
has_bbox, accurate_localizations, dice_scores = keras_preds.process_prediction(config,
                                                                               predictions_unique_name,
                                                                               predictions_path,
                                                                               r=r[pooling_operator],
                                                                               pool_method=pool_dict[pooling_operator],
                                                                               img_pred_method=image_prediction_method,
                                                                               threshold_binarization=0.5,
                                                                               iou_threshold=0.1)


keras_preds.save_generated_files(predictions_path, predictions_unique_name, image_labels, image_predictions,
                                 has_bbox, accurate_localizations, dice_scores)
keras_preds.save_results_table(image_prediction_method, image_labels, image_predictions, class_name, predictions_unique_name,
                               performance_path, has_bbox, accurate_localizations, dice_scores)