import numpy as np
import tensorflow as tf
import random as rn
import os
import yaml
import argparse
from cnn.cross_validation import cross_validation

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

cross_validation(config, number_splits=5)
