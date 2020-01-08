import yaml
import argparse
import os
import tensorflow as tf

from cnn.performance_eval import performance_evaluation

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

dataset_name = 'test_set_Cardiomegaly_CV0'
performance_evaluation(config, dataset_name, pool_method='mean', image_prediction_method='as_production',
                       th_binarization=0.5, th_iou=0.1)