import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from keras.preprocessing import image
import os
from pathlib import Path
from keras.applications.resnet50 import preprocess_input
import cv2
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
import imagesize

from keras_utils import visualize_population
np.random.seed(0)
FINDINGS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
            'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
            'Pneumonia', 'Pneumothorax']

LOCALIZATION = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass',
                'Nodule', 'Pneumonia', 'Pneumothorax']

IMAGE_X = 512
IMAGE_Y = 512
PATCH_SIZE = 16


def load_csv(file_path):
    print('Loading data ...')

    bbox = pd.read_csv(file_path)
    print('Cardiomegaly label division')
    print(bbox['Cardiomegaly'].value_counts())
    return bbox.dropna(axis=1)


def rename_columns(df, classification_csv = True):
    if classification_csv:
        return df.rename(columns={'OriginalImage[Width': 'Width', 'Height]': 'Height',
                              'OriginalImagePixelSpacing[x': 'PixelSpacing_x', 'y]': 'PixelSpacing_y'})
    else:
        return df.rename(columns={'Bbox [x': 'x', 'h]': 'h'})


def get_label_by_imageind(label_df, image_ind):
    return label_df.loc[label_df['Image Index']== image_ind]


def add_label_columns(df):
    df['Bbox'] = pd.Series(0, index=df.index).astype(int)
    for label in FINDINGS:
        new_column = df['Finding Labels'].str.contains(label)
        # add new column and fill in with result above and attach to the initial df
        df[label]= pd.Series(new_column, index=df.index).astype(float)
        df[label+'_loc'] = pd.Series(0, index=df.index).astype(float)
    return df


def get_ann_list(df):
    return df['Image Index']


## TODO: to remove - not used currently
def converto_3_color_channels(img):
    stacked_img = np.stack((img,) * 3, axis=-1)
    return stacked_img


def get_classification_labels(label_dir, preprocessed_csv=False):
    Y = load_csv(label_dir)
    # TODO: turn into new preparation function
    if not preprocessed_csv:
        Y_pr = rename_columns(Y, True)
        Y = add_label_columns(Y_pr)
    return Y


def drop_extra_label_columns(df):
    # dropping 'Image Index', 'Patient ID', Image Found and ReorderIndex
    return df[[ 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
         'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
         'Pneumonia', 'Pneumothorax']]


def process_image(img_path):
    img = image.load_img(img_path, target_size=(512, 512))
    x = image.img_to_array(img)
    return preprocess_input(x)


def scaling_factor_v2(img_path):
    x, y = imagesize.get(img_path)
    return x/IMAGE_X, y/IMAGE_Y


def reorder_rows(df):
    return df.sort_values(by=["Reorder Index"])


def preprocess_labels(Yclass, path_to_png):
    xy_df = Yclass.copy(deep=True)
    xy_df['Image Found'] = None
    xy_df['Reorder Index'] = None
    xy_df['Dir Path'] = None

    for src_path in Path(path_to_png).glob('**/*.png'):
        image_ind = os.path.basename(src_path)
        xy_df.loc[xy_df['Image Index'] == image_ind, ['Dir Path']] = str(src_path)
    print("xy before dropping")
    print(xy_df.shape)
    xy_df = xy_df.dropna(subset=['Dir Path'])
    print(xy_df.shape)
    return reorder_rows(xy_df)


def translate_on_patches(x_min, y_min, x_max, y_max):
    x = int(np.round((x_min/IMAGE_X)*PATCH_SIZE))
    y = int(np.round((y_min/IMAGE_Y)*PATCH_SIZE))
    x_max = int(np.round((x_max/IMAGE_X)*PATCH_SIZE))
    y_max = int(np.round((y_max/IMAGE_Y)*PATCH_SIZE))
    return x, y, x_max, y_max


def couple_location_labels(Y_loc_dir, Y_class, P, out_dir):
    Y_loc = rename_columns(load_csv(Y_loc_dir), False)

    for diagnosis in FINDINGS:
        Y_class[[diagnosis + '_loc','Bbox']]= Y_class.apply(lambda x: pd.Series(integrate_annotations(x, Y_loc, diagnosis, P)), axis=1)

    Y_class.to_csv(out_dir+'/processed_Y.csv')
    return Y_class


def create_label_matrix_classification(row, label, P):
    if row[label]==1:
        im_q = np.ones((P, P), np.float)
        return im_q
    else:
        im_q = np.zeros((P, P), np.float)
        return im_q


def make_label_matrix_localization_v2(P, x_min, y_min, x_max, y_max):
    im_q = np.zeros((P, P), np.float)
    im_q[y_min:(y_max + 1), x_min:(x_max + 1)] = 1.
    return im_q

def get_all_bbox_for_image(row, Y_loc):
    all_info = Y_loc.loc[Y_loc['Image Index'] == row['Image Index']]
    return all_info, row


def integrate_annotations(row, Y_loc, diagnosis, P):
    result_image_class = []
    all_rows, row_classif_df = get_all_bbox_for_image(row, Y_loc)

    # if no bbox is found for this image
    if all_rows.values.size==0:
        y_mat = create_label_matrix_classification(row, diagnosis, P)

        result_image_class.append(y_mat)
        return [y_mat, 0]
    else:

        if diagnosis in all_rows['Finding Label'].values:
            # y_mat = all_rows.groupby(['Image Index']).apply(lambda x: image_with_bbox_v2(x, row_classif_df, diagnosis, P))
            y_mat = all_rows.apply(lambda x: create_label_matrix_localization(x, row_classif_df, diagnosis, P), axis=1)

            result_image_class.append(y_mat.dropna().values[0])
            return [y_mat.dropna().values[0], 1]
        else:
            y_mat = create_label_matrix_classification(row, diagnosis, P)
            result_image_class.append(y_mat)

            return [y_mat, 1]


def create_label_matrix_localization(row, row_classif_df, diagnosis, P):
    if row.values.size > 0:
        if diagnosis == row['Finding Label']:
            # scale_x, scale_y = scaling_factor(row_classif_df['Dir Path'])
            scale_x, scale_y = scaling_factor_v2(row_classif_df['Dir Path'])
            x_min, y_min, x_max, y_max = translate_coords_to_new_image_size(row['x'], row['y'], row['w'], row['h'],
                                                                            scale_x,
                                                                            scale_y)
            x_min, y_min, x_max, y_max = translate_on_patches(x_min, y_min, x_max, y_max)
            #y_mat = make_label_matrix_localization(PATCH_SIZE, x_min, y_min, x_max, y_max)
            y_mat = make_label_matrix_localization_v2(PATCH_SIZE, x_min, y_min, x_max, y_max)

            return y_mat
    else:
        print("this hsould NOT BE PRINTING ")


def translate_coords_to_new_image_size(x, y, w, h, scale_x, scale_y):
    x_min = x/ scale_x
    y_min = y / scale_y
    x_max = x_min + w / scale_x
    y_max = y_min + h / scale_y
    return x_min, y_min, x_max, y_max


def bbox_available(df, img_ind):
    return df.loc[df['Image Index'] == img_ind, ['Bbox']]


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


def split_test_train(X, Y, test_ratio=0.2):
    # shuffle split ensuring that same patient ID is only in test or train
    train_inds, test_inds = next(GroupShuffleSplit(test_size=test_ratio, random_state=0).split(Y, groups=Y['Patient ID']))
    Y = drop_extra_label_columns(Y)

    X_train = np.take(X, train_inds, axis=0)
    X_test = np.take(X, test_inds, axis=0)
    return X_train, X_test, Y.iloc[train_inds], Y.iloc[test_inds]


def split_test_train_v2(df, test_ratio=0.2, random_state=None):
    # shuffle split ensuring that same patient ID is only in test or train
    train_inds, test_inds = next(GroupShuffleSplit(test_size=test_ratio, random_state=random_state).split(df, groups=df['Patient ID']))

    return train_inds, test_inds, df.iloc[train_inds], df.iloc[test_inds]


def split_test_train_stratified(df, test_ration, random_state = None):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)



def separate_localization_classification_labels(Y):
    return Y.loc[Y['Bbox']==0], Y.loc[Y['Bbox']==1]


# THIS METHOD IS USED FOR KERAS TESTING
def keep_index_and_diagnose_columns(Y):
    return Y[['Dir Path', 'Atelectasis_loc', 'Cardiomegaly_loc', 'Consolidation_loc', 'Edema_loc',
        'Effusion_loc', 'Emphysema_loc','Fibrosis_loc', 'Hernia_loc', 'Infiltration_loc', 'Mass_loc',
        'Nodule_loc', 'Pleural_Thickening_loc', 'Pneumonia_loc', 'Pneumothorax_loc']]


def keep_index_and_1diagnose_columns(Y):
    return Y[['Dir Path', 'Consolidation_loc']]


# Lastly, We use 80% annotated images and 50% unanno-tated images to train the model and evaluate
#  on the other 20% annotated images in each fold.
def get_train_test(Y, random_state=None, do_stats=False, res_path =None):
    classification, bbox = separate_localization_classification_labels(Y)

    _, _, df_class_train, df_class_test = split_test_train_v2(classification, test_ratio=0.5, random_state=random_state)
    train_bbox_idx, _, df_bbox_train, df_bbox_test = split_test_train_v2(bbox, test_ratio=0.2, random_state=random_state)
    print("BBO TRAIN")
    print(df_bbox_train.shape)
    print(df_bbox_test.shape)

    train_clas_idx, _, df_class_train, df_class_val = split_test_train_v2(df_class_train, test_ratio=0.2, random_state=random_state)

    train_idx = np.concatenate((train_clas_idx, train_bbox_idx), axis=None)
    df_train = pd.concat([df_class_train, df_bbox_train])
    df_val = df_class_val

    if do_stats and res_path is not None:
        visualize_population(Y, 'whole_df_group', res_path, FINDINGS)
        visualize_population(df_train, 'train_group', res_path, FINDINGS)
        visualize_population(df_val, 'validation_group', res_path, FINDINGS)
        visualize_population(df_bbox_test, 'test_bbox_group', res_path, FINDINGS)
        visualize_population(df_class_test, 'test_class_group', res_path, FINDINGS)
        visualize_population(pd.concat([df_bbox_test, df_class_test]), 'test_group', res_path, FINDINGS)

    train_set, val_set = keep_index_and_1diagnose_columns(df_train), keep_index_and_1diagnose_columns(df_val)
    bbox_test, class_test = keep_index_and_1diagnose_columns(df_bbox_test), keep_index_and_1diagnose_columns(df_class_test)
    bbox_train = keep_index_and_1diagnose_columns(df_bbox_train)

    return train_idx, train_set, val_set, bbox_test, class_test, bbox_train




