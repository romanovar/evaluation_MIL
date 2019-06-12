from keras.applications import ResNet50V2
from keras.layers import Input
from keras.preprocessing import image
import numpy as np
from keras.applications.resnet50 import preprocess_input
import os
from pathlib import Path

from ReadXray import get_label_by_imageind


def process_image(img_path):
    img = image.load_img(img_path, target_size=(512, 512))
    x = image.img_to_array(img)
    print(x.shape)
    return preprocess_input(x)


def load_process_png(label_df, path_to_png="C:/Users/s161590/Desktop/Data/X_Ray/images"):
    """
    Searches recursively for png files starting from the common parent dir
    When png files is found, it is loaded and added to the final list
    :param path_to_png: common parent directory path for all dicom files
    :return: list with all loaded dicom files found
    """
    png_files = []
    labels = []
    for src_path in Path(path_to_png).glob('**/*.png'):
        image_ind = os.path.basename(src_path)

        Y = get_label_by_imageind(label_df, image_ind)
        png_files.append(process_image(src_path))
        # drop column with img ind in Y
        Y = Y.iloc[:, 1:15]
        labels.append(np.array(Y.values))
    return np.array(png_files), np.array(labels)


def get_process_annotated_png(ann_list, path_to_png="C:/Users/s161590/Desktop/Data/X_Ray/images"):
    """
    Searches recursively for png files starting from the common parent dir
    When png files is found, it is loaded and added to the final list
    :param path_to_png: common parent directory path for all dicom files
    :return: list with all loaded dicom files found
    """
    png_files = []
    for src_path in Path(path_to_png).glob('**/*.png'):
        image_ind = os.path.basename(src_path)
        for img in ann_list:
            #tODO: should NOT only load these files --> currently is a test purpose
            if img == image_ind:
                png_files.append(process_image(src_path))
    print("Annotated images found: " + str(np.array(png_files).shape))
    return np.array(png_files)


def get_feature_extraction(x):
    input_tensor = Input(shape=(512, 512, 3))
    model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(512, 512, 3), pooling=None)
    fe_mat = model.predict(x)
    # print(fe_mat.shape)
    return fe_mat



# img_path ="C:/Users/s161590/Desktop/Data/X_Ray/images/00000001_000.png"
# img = image.load_img(img_path, target_size=(512, 512))
# x = image.img_to_array(img)
# print(x)
# x = np.expand_dims(x, axis=0)
# print(x.shape)
# x = preprocess_input(x)
# # print(x.shape)
# preds = model.predict(x)
# print(type(preds))
#
