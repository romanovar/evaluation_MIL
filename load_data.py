import pandas as pd
import numpy as np
from keras.preprocessing import image
import os
from pathlib import Path
from keras.applications.resnet50 import preprocess_input


FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
            'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
            'Pneumonia', 'Pneumothorax']


def load_csv(file_path):
    bbox = pd.read_csv(file_path)
    return bbox.dropna(axis=1)


def preprocess_classification_csv(df):
    return df.rename(columns={'OriginalImage[Width': 'Width', 'Height]': 'Height',
                              'OriginalImagePixelSpacing[x': 'PixelSpacing_x', 'y]': 'PixelSpacing_y'}, inplace=True)


def get_label_by_imageind(label_df, image_ind):
    return label_df.loc[label_df['Image Index']== image_ind]


def add_label_columns(df):
    for label in FINDINGS:
        new_column = df['Finding Labels'].str.contains(label)
        # add new column and fill in with result above and attach to the initial df
        df[label]= pd.Series(new_column, index=df.index)
    return df


def get_ann_list(df):
    return df['Image Index']


## TODO: to remove - not used currently
def converto_3_color_channels(img):
    stacked_img = np.stack((img,) * 3, axis=-1)
    return stacked_img


# #Todo: loads currently only
# def find_load_annotated_png_files(image_ind_with_bbox, path_to_png="C:/Users/s161590/Desktop/Data/X_Ray/images"):
#     """
#     Searches recursively for dicom files starting from the common parent dir
#     When dicom files is found, it is loaded and added to the final list
#     :param path_to_png: common parent directory path for all dicom files
#     :return: list with all loaded dicom files found
#     """
#     png_files = []
#     for src_path in Path(path_to_png).glob('**/*.png'):
#         image_ind = os.path.basename(src_path)
#         for img in image_ind_with_bbox:
#             if img == image_ind:
#                 png = cv2.imread(os.path.join(os.path.dirname(src_path), os.path.basename(src_path)),0-1)
#                 # img = cv2.resize(png, (224, 224))
#
#                 png_files.append(converto_3_color_channels(png))
#
#     print("Annotated images found: " + str(np.array(png_files).shape))
#     return np.array(png_files)
#
#
# def find_load_png_files(path_to_png="C:/Users/s161590/Desktop/Data/X_Ray/images"):
#     """
#     Searches recursively for dicom files starting from the common parent dir
#     When dicom files is found, it is loaded and added to the final list
#     :param path_to_png: common parent directory path for all dicom files
#     :return: list with all loaded dicom files found
#     """
#     png_files = []
#     for src_path in Path(path_to_png).glob('**/*.png'):
#         image_ind = os.path.basename(src_path)
#         # for img in image_ind_with_bbox:
#         #     if img == image_ind:
#         #         print("Annotations of image found: "+ str(image_ind))
#         png = cv2.imread(os.path.join(os.path.dirname(src_path), os.path.basename(src_path)),-1)
#         img = cv2.resize(png, (224, 224))
#         png_files.append(img)
#     return png_files
#

def get_classification_labels(label_dir="C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv"):
    data_labels = load_csv(label_dir)
    Y = add_label_columns(data_labels)
    # SAVE CSV
    # Y.to_csv("C:/Users/s161590/Desktop/Data/X_Ray/processed_Y.csv")

    Y = Y.iloc[:, [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
    return Y


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


bbox = load_csv("C:/Users/s161590/Desktop/Data/X_Ray/BBox_List_2017.csv")

image_ind_with_bbox = get_ann_list(bbox)
# find_load_annotated_png_files(image_ind_with_bbox, path_to_png="C:/Users/s161590/Desktop/Data/X_Ray/images")

