from pathlib import Path
import os

from sklearn.model_selection import StratifiedShuffleSplit

from cnn.preprocessor.load_data import keep_index_and_1diagnose_columns
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
    return df.iloc[train_inds], df.iloc[test_inds]


def load_pascal(pascal_img_path):
    df = create_csv(pascal_img_path)
    df.to_csv(pascal_img_path+'/pascal_data.csv')

    train_val, test = get_train_test_1fold(df)
    train, val = get_train_test_1fold(train_val)

    df_train = keep_index_and_1diagnose_columns(train, 'Instance labels')
    df_test = keep_index_and_1diagnose_columns(test, 'Instance labels')
    df_val = keep_index_and_1diagnose_columns(val, 'Instance labels')
    return df_train, df_val, df_test


