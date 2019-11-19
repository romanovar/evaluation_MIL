import yaml
import argparse

from stability.stability_all_classifiers.multi_classifier_stability import stability_all_classifiers_instance_level,\
    stability_all_classifiers_bag_level


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)


set_name1 = 'subset_test_set_CV0_0_0.95.npy'
set_name2='subset_test_set_CV0_1_0.95.npy'
set_name3 = 'subset_test_set_CV0_2_0.95.npy'
set_name4='subset_test_set_CV0_3_0.95.npy'
set_name5='subset_test_set_CV0_4_0.95.npy'
classifiers = [set_name1, set_name2, set_name3, set_name4, set_name5]
path = 'C:/Users/s161590/Documents/Project_li/bbox_images/'

# stability_all_classifiers_instance_level(config, classifiers, only_segmentation_images=True, only_positive_images=False,
#                                          visualize_per_image= True)
stability_all_classifiers_instance_level(config, classifiers, only_segmentation_images=True, only_positive_images=False,
                                         visualize_per_image= True)
# stability_all_classifiers_bag_level(config, classifiers, only_segmentation_images=False, only_positive_images=True)
# stability_all_classifiers_bag_level(config, classifiers, only_segmentation_images=False, only_positive_images=False)
# stability_all_classifiers(config, classifiers, only_segmentation_images=False)