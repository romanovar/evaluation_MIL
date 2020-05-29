import argparse
import yaml
import pandas as pd
import cnn.preprocessor.load_data_datasets as ldd
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array
from cnn.preprocessor.load_data_mura import padding_needed, pad_image
from cnn.keras_utils import process_loaded_labels, image_larger_input, calculate_scale_ratio
from cnn.preprocessor.process_input import decrease_image_size, preprocess_images_from_dataframe, \
    combine_preprocessed_csv


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

use_xray_dataset = config['use_xray_dataset']
use_pascal_dataset = config['use_pascal_dataset']
mura_interpolation = config['mura_interpolation']
image_path = config['image_path']
resized_images_before_training = config['resized_images_before_training']
processed_labels_path=config['processed_labels_path']

IMAGE_SIZE = 512

if use_xray_dataset:
    df_xray = ldd.load_process_xray14(config)

xray_df = pd.read_csv(processed_labels_path).copy()

## currently only working for Xray dataset
if resized_images_before_training:
    df_processed, xray_df = preprocess_images_from_dataframe(df_xray, IMAGE_SIZE, IMAGE_SIZE, mura_interpolation, image_path,
                                                'processed_imgs', xray_df)

    xray_df.to_csv(image_path+'/all_processed_images.csv')
