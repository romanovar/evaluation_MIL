import yaml
import argparse
import os
import tensorflow as tf
import keras_preds
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

skip_processing = config['skip_processing_labels']
image_path = config['image_path']
classication_labels_path = config['classication_labels_path']
localization_labels_path = config['localization_labels_path']
results_path = config['results_path']
processed_labels_path = config['processed_labels_path']
train_mode = config['train_mode']
test_single_image = config['test_single_image']
prediction_skip_processing = config['prediction_skip_processing']
predict_res_path = config['prediction_results_path']




#########################################################


################# STEP 1 ###########################
# predict_res_path = 'C:/Users/s161590/Desktop/Project_li/single_class/5050/'

dataset_name = 'single_5050_train_set'
image_prediction_method = 'as_loss'
# keras_preds.process_prediction_v2(dataset_name, predict_res_path, img_pred_as_loss=image_prediction_method,
#                                   batch_size=10)
# image_prediction_method2 = 'as_production'
# keras_preds.process_prediction_v2_image_level(dataset_name, predict_res_path, img_pred_as_loss=image_prediction_method2,
#                                               batch_size=10)

################# STEP 2 ###########################
dataset_name = 'single_patient_train_set'
image_prediction_method = 'as_loss'
# image_prediction_method2 = 'as_production'
# predict_res_path = 'C:/Users/s161590/Desktop/Project_li/predictions/'

keras_preds.combine_auc_accuracy_1class(dataset_name, image_prediction_method, predict_res_path)
