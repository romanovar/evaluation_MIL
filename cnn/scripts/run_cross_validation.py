
import yaml
import argparse
import os

from cnn.cross_validation import cross_validation

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

cross_validation(config)
