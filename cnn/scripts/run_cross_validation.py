from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import os
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
import yaml
import argparse

from cnn.cross_validation import cross_validation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

cross_validation(config)
