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

#dataset_name = 'test_set'
#image_prediction_method = 'as_loss'
#do_predictions_set(dataset_name, prediction_skip_processing, image_prediction_method)

################# STEP 1 ###########################
dataset_name = 'single_5050'
image_prediction_method = 'as_loss'
keras_preds.process_prediction_v2(dataset_name, predict_res_path, img_pred_as_loss=image_prediction_method,
                                  batch_size=10)
image_prediction_method2 = 'as_production'
keras_preds.process_prediction_v2_image_level(dataset_name, predict_res_path, img_pred_as_loss=image_prediction_method2,
                                              batch_size=10)

################# STEP 2 ###########################
# dataset_name = '100_e17train_set'
# image_prediction_method = 'as_production'
# keras_preds.combine_npy_accuracy(dataset_name, results_path)
# keras_preds.combine_npy_auc(dataset_name, image_prediction_method, results_path)

#dataset_name = 'val_set'
#image_prediction_method = 'as_loss'
#do_predictions_set(dataset_name, prediction_skip_processing, image_prediction_method)

#################################################################
#dataset_name = 'test_set'
#image_prediction_method = 'as_production'
#do_predictions_set(dataset_name, prediction_skip_processing, image_prediction_method)

#
# dataset_name = 'train_set'
# image_prediction_method = 'as_production'
# do_predictions_set(dataset_name, prediction_skip_processing, image_prediction_method)

#dataset_name = 'val_set'
#image_prediction_method = 'as_production'
#do_predictions_set(dataset_name, prediction_skip_processing, image_prediction_method)
############################################################3




############################# TEST SET ########################

# auc_v1_test, auc_v2_test, auc_v3_test, acc_test = process_prediction('test_set', results_path)

# auc_v1_val, auc_v2_val, auc_v3_val, acc_val = process_prediction('val_set', results_path)

# auc_v1_train, auc_v2_train, auc_v3_train, acc_train = process_prediction('train_set', results_path)



# keras_utils.plot_grouped_bar_auc(auc_v1_train, auc_v1_val, auc_v1_test, 'auc_v1', results_path, ld.FINDINGS)
# keras_utils.plot_grouped_bar_auc(auc_v2_train, auc_v2_val, auc_v2_test, 'auc_v2', results_path, ld.FINDINGS)
# keras_utils.plot_grouped_bar_auc(auc_v3_train, auc_v3_val, auc_v3_test, 'auc_v3', results_path, ld.FINDINGS)
#
# keras_utils.plot_grouped_bar_accuracy(acc_train, acc_val, acc_test, 'accuracy', results_path, ld.FINDINGS)