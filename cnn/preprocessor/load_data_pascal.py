from pathlib import Path
import os

from sklearn.model_selection import StratifiedShuffleSplit

from cnn.preprocessor.load_data import keep_index_and_1diagnose_columns, get_rows_from_indices
from cnn.preprocessor.load_data_mura import create_instance_labels
import pandas as pd


def create_csv(path_to_png):
    df = pd.DataFrame()
    df['Dir Path'] = None
    df['Label'] = None
    df['Instance labels'] = None
    images_path_list = []
    labels_list = []
    instance_labels = []

    for src_path in Path(path_to_png).glob('**/*.png'):
        image_name = os.path.basename(src_path)
        images_path_list.append(str(src_path))
        bag_label = int(image_name.__contains__('cars'))
        label_string = image_name.split('_')[0]
        # labels_list.append(bag_label)
        labels_list.append(label_string)
        instance_labels.append(create_instance_labels(bag_label, 16))
    df['Dir Path'] = images_path_list
    df['Label'] = labels_list
    df['Instance labels'] = instance_labels
    return df


def get_train_test_1fold(df):
    train_inds, test_inds = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(df, df['Label']))
    assert train_inds.all() != test_inds.all(), "overlapp occur"
    return df.iloc[train_inds], df.iloc[test_inds]


def load_pascal(pascal_img_path):
    df = create_csv(pascal_img_path)
    df.to_csv(pascal_img_path+'/pascal_data.csv')
    return df


def split_train_val_test(df):
    train_val, test = get_train_test_1fold(df)
    train, val = get_train_test_1fold(train_val)

    df_train = keep_index_and_1diagnose_columns(train, 'Instance labels')
    df_test = keep_index_and_1diagnose_columns(test, 'Instance labels')
    df_val = keep_index_and_1diagnose_columns(val, 'Instance labels')
    return df_train, df_val, df_test


def split_data_cv(df, nr_cv):
    sss = StratifiedShuffleSplit(n_splits=nr_cv, random_state=0)
    train_ind_col = []
    test_ind_col = []
    for train_inds, test_inds in sss.split(df, df['Label']):
        assert train_inds.all() != test_inds.all(), "overlapp occur"
        train_ind_col.append(train_inds)
        test_ind_col.append(test_inds)
    return train_ind_col, test_ind_col


def construct_train_test_cv(df, nr_cv, split):
    train_val_ind_col, test_ind_col = split_data_cv(df, nr_cv)
    df_train_val, df_test = get_rows_from_indices(df, train_val_ind_col[split], test_ind_col[split])
    train_ind_col, val_ind_col = split_data_cv(df_train_val, nr_cv)
    df_train, df_val = get_rows_from_indices(df_train_val, train_ind_col[split], val_ind_col[split])

    train_set = keep_index_and_1diagnose_columns(df_train, 'Instance labels')
    val_set = keep_index_and_1diagnose_columns(df_val, 'Instance labels')
    test_set = keep_index_and_1diagnose_columns(df_test, 'Instance labels')

    return train_set, val_set, test_set