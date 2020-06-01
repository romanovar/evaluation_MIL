import argparse
import os

import tensorflow as tf
import yaml

from cnn import keras_preds

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

predict_res_path = config['prediction_results_path']
class_name = config['class_name']
use_xray = config['use_xray_dataset']

image_prediction_method = 'as_production'
predictions_unique_name = 'car_test_set_CV1_4car_0.95'
pool_method = 'nor'
r = 0.1

image_labels, image_predictions, \
has_bbox, accurate_localizations, dice_scores, inst_auc = keras_preds.process_prediction(config,
                                                                                         predictions_unique_name,
                                                                                         predict_res_path,
                                                                                         r=r,
                                                                                         pool_method=pool_method,
                                                                                         img_pred_method=image_prediction_method,
                                                                                         threshold_binarization=0.5,
                                                                                         iou_threshold=0.1)


keras_preds.save_generated_files(predict_res_path, predictions_unique_name, image_labels, image_predictions,
                                 has_bbox, accurate_localizations, dice_scores)

if use_xray:
    keras_preds.compute_save_accuracy_results(predictions_unique_name, predict_res_path, has_bbox, accurate_localizations)
    keras_preds.compute_save_dice_results(predictions_unique_name, predict_res_path, has_bbox, dice_scores)
    keras_preds.compute_save_auc(predictions_unique_name, image_prediction_method, predict_res_path,
                                 image_labels, image_predictions, class_name)
else:
    keras_preds.compute_save_auc(predictions_unique_name, image_prediction_method, predict_res_path,
                                 image_labels, image_predictions, class_name)
