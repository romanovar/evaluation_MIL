import pandas as pd
import cv2
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import numpy as np
import math

from cnn.preprocessor.load_data import keep_index_and_1diagnose_columns, calculate_observations_to_keep

CLASS_LIST = ['elbow', 'finger', 'forearm', 'hand', 'humerus', 'shoulder', 'wrist']


def check_validity_class(class_name):
    assert class_name in CLASS_LIST, "Ensure specified class is right"


def read_csv_add_columns(file_path, column_list_names = None, preprocessed_csv=False):
    print('Loading data ...')
    if column_list_names is None:
        df = pd.read_csv(file_path, header=None)
    else:
        df = pd.read_csv(file_path, names=column_list_names)
    if not preprocessed_csv:
        df['class'] = 0
        df['label'] = 0
        df['instance labels'] = 0
    return df


def load_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    return df


def create_instance_labels(bag_label, P):
    if bag_label==1:
        im_q = np.ones((P, P), np.float)
        # str(test_str).replace('\n', '')
        return str(im_q).replace('\n', '')
        # return im_q

    else:
        im_q = np.zeros((P, P), np.float)
        # return im_q
        return str(im_q).replace('\n', '')


def combine_labels_and_path(df_labels, df_img_path, file_path_root, csv_name):
    for index, row in tqdm(df_labels.iterrows()):
        file_path_substring = row[0]
        file_label_bag = row[1]

        if df_img_path.iloc[:, 0].str.contains(file_path_substring).any():
            # print(file_path)
            matching_indices = df_img_path.index[df_img_path.iloc[:, 0].str.contains(file_path_substring)]
            ### UPDATE PATH TO IMAGES
            file_path_in_csv = df_img_path.loc[matching_indices, 'Dir Path']
            df_img_path.loc[matching_indices, 'Dir Path'] = file_path_root + file_path_in_csv
            start_class = file_path_substring.find('XR_') + 3
            end_class = file_path_substring.find('patient', start_class)
            class_present = file_path_substring[start_class:end_class - 1]
            df_img_path.loc[matching_indices, 'class'] =  class_present.lower()
            df_img_path.loc[matching_indices, 'label'] =file_label_bag
            df_img_path.loc[matching_indices, 'instance labels'] = create_instance_labels(file_label_bag, 16)
    df_img_path.to_csv(file_path_root+ 'MURA-v1.1/' + csv_name+'.csv')
    return df_img_path


def load_mura(skip_processing, processed_train_labels_path, processed_test_labels_path,
              mura_train_img_path, mura_train_labels_path,
              mura_test_labels_path, mura_test_img_path):
    if skip_processing:
        df_train_val = pd.read_csv(processed_train_labels_path)
        test_df_all_classes = pd.read_csv(processed_test_labels_path)

    else:
        end_class = mura_train_img_path.find('MURA-v1.1')
        mura_folder_root = mura_train_img_path[0:end_class]
        print(mura_folder_root)
        df_train_val = get_save_processed_df(mura_train_labels_path, mura_train_img_path, mura_folder_root, "train_mura")
        test_df_all_classes = get_save_processed_df(mura_test_labels_path, mura_test_img_path, mura_folder_root,
                                                        "test_mura")
    return df_train_val, test_df_all_classes


def prepare_mura_set(df_train_val, test_df_all_classes, class_name):
    _, _, train_df_all_classes, val_df_all_classes = split_train_val_set(df_train_val)
    df_train, df_val, df_test = filter_all_set_for_class(train_df_all_classes, val_df_all_classes,
                                                             test_df_all_classes, class_name)
    df_train_final = keep_index_and_1diagnose_columns(df_train, 'instance labels')
    df_val_final = keep_index_and_1diagnose_columns(df_val, 'instance labels')
    df_test_final = keep_index_and_1diagnose_columns(df_test, 'instance labels')

    print('Training set: ' + str(df_train_final.shape))
    print('Validation set: ' + str(df_val_final.shape))
    # print('Localization testing set: '+ str(df_bbox_test.shape))
    print('Classification testing set: ' + str(df_test_final.shape))
    return df_train_final, df_val_final, df_test_final


def get_save_processed_df(labels_df_path, img_paths_df_path, file_path, csv_name):
    img_paths_df = read_csv_add_columns(img_paths_df_path, column_list_names=['Dir Path'])
    img_labels_df =pd.read_csv(labels_df_path, header=None)
    return combine_labels_and_path(img_labels_df, img_paths_df, file_path, csv_name=csv_name)


def split_train_val_set(df):
    print("Splitting data ...")
    train_inds, test_inds = next(ShuffleSplit(n_splits=1, random_state=0, test_size=0.2).split(df))
    return train_inds, test_inds, df.iloc[train_inds], df.iloc[test_inds]


def split_test_train_cv(df, splits_nr, test_ratio=0.2, random_state=None):
    # shuffle split ensuring that same patient ID is only in test or train
    gss = ShuffleSplit(n_splits=splits_nr, test_size=test_ratio, random_state=random_state)
    splits_iter = gss.split(df)

    train_ind_col = []
    test_ind_col = []
    for split in splits_iter:
        train_inds, test_inds = split
        train_ind_col.append(train_inds)
        test_ind_col.append(test_inds)
    return train_ind_col, test_ind_col


def split_data_cv(df, splits_nr, current_split, random_seed,  diagnose_col, ratio_to_keep=None):
    df_train_val = filter_rows_on_class(df, class_name=diagnose_col)
    train_inds_coll, val_inds_coll = split_test_train_cv(df_train_val, splits_nr, test_ratio=0.2, random_state=random_seed)
    df_train = df.iloc[train_inds_coll[current_split]]
    df_val = df.iloc[val_inds_coll[current_split]]
    df_train_final = keep_index_and_1diagnose_columns(df_train, 'instance labels')
    df_val_final = keep_index_and_1diagnose_columns(df_val, 'instance labels')
    return df_train_final, df_val_final


def get_train_subset_mura(train_set, random_seed, ratio_to_keep):
    obs_to_keep = np.ceil(ratio_to_keep * len(train_set))

    if obs_to_keep > 0:
        np.random.seed(seed=random_seed)
        class_train_ind_keep = np.random.choice(train_set.index, int(obs_to_keep), replace=False)
        train_subset = train_set.loc[class_train_ind_keep]
        return train_subset
    else:
        return train_set


def filter_rows_and_columns(df, class_name):
    df = filter_rows_on_class(df, class_name=class_name)
    return keep_index_and_1diagnose_columns(df, 'instance labels')


def filter_rows_on_class(df, class_name):
    return df[(df['class'] == class_name)]


def filter_all_set_for_class(train_df, val_df, test_df, class_name):
    train = filter_rows_on_class(train_df, class_name)
    test = filter_rows_on_class(test_df, class_name)
    valid = filter_rows_on_class(val_df, class_name)
    return train, valid, test


def padding_needed(img):
    assert img.shape[0] <= 512, "x axis is bigger than 512 pixels"
    assert img.shape[1] <= 512, "y axis is bigger than 512 pixels"
    if img.shape[0] == 512 and img.shape[1] == 512:
        return False
    else:
        return True


def pad_image(img, final_size_x, final_size_y):
    bgr_color_padding = [0, 0, 0]
    pad_left_right = (final_size_x - img.shape[1])/2
    pad_bottom_top = (final_size_y - img.shape[0])/2
    constant = cv2.copyMakeBorder(img, borderType=cv2.BORDER_CONSTANT, top=math.ceil(pad_bottom_top),
                                  bottom=math.floor(pad_bottom_top),
                                  left=math.ceil(pad_left_right), right=math.floor(pad_left_right),
                                  value=bgr_color_padding)
    assert constant.shape[0] == final_size_x and constant.shape[1]==final_size_y, "error during padding an image"
    return constant
