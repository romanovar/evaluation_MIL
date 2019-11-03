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

performance_evaluation(config, "test_setBLL", batch_size=1)