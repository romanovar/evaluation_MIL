import numpy as np
import tensorflow as tf
import random as rn
import yaml
import argparse
import os

from cnn.subsets_training import train_on_subsets

np.random.seed(1)
tf.random.set_seed(2)
rn.seed(1)
os.environ['PYTHONHASHSEED']='1'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

overlap_ratio = 0.95
CV_SPLITS = 5
number_classifiers = 5
# this list should have the same length as the number of classifiers
subset_seeds = [1234, 5678, 9012, 3456, 7890]

train_on_subsets(config, number_splits=CV_SPLITS, CV_split_to_use=1, number_classifiers=number_classifiers,
                 subset_seeds=subset_seeds, overlap_ratio=overlap_ratio)
