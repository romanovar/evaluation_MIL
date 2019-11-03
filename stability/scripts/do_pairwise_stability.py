import yaml
import argparse

from stability.stability_2classifiers.analysis_stability_2classifiers import calculate_vizualize_save_stability


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

set_name1 = 'subset_test_set_CV0_2_0.95.npy'
set_name2='subset_test_set_CV0_4_0.95.npy'

calculate_vizualize_save_stability(set_name1, set_name2, config)
