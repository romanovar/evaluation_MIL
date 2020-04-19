import pandas as pd
import numpy as np
from keras.preprocessing import image
import os
from pathlib import Path
from keras.applications.resnet50 import preprocess_input
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
import imagesize

from cnn.keras_utils import visualize_population
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
    df = pd.read_csv(file_path)
    return df.dropna(axis=1)


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

    Y_class.to_csv(out_dir+'/processed_new_Y.csv')
    return Y_class


def create_label_matrix_classification(row, label, P):
    if row[label]==1:
        im_q = np.ones((P, P), np.float)
        # str(test_str).replace('\n', '')
        return str(im_q).replace('\n', '')
        # return im_q

    else:
        im_q = np.zeros((P, P), np.float)
        # return im_q
        return str(im_q).replace('\n', '')

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
            # return [np.array2string(y_mat.dropna().values[0], separator=' '), 1]
            return [str(y_mat.dropna().values[0]).replace('\n', ''), 1]

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

#
# def split_test_train(X, Y, test_ratio=0.2):
#     # shuffle split ensuring that same patient ID is only in test or train
#     train_inds, test_inds = next(GroupShuffleSplit(test_size=test_ratio, random_state=0).split(Y, groups=Y['Patient ID']))
#     Y = drop_extra_label_columns(Y)
#
#     X_train = np.take(X, train_inds, axis=0)
#     X_test = np.take(X, test_inds, axis=0)
#     return X_train, X_test, Y.iloc[train_inds], Y.iloc[test_inds]

# shuffle split ensuring that same patient ID is only in test or train


def split_test_train_v2(df, test_ratio=0.2, random_state=None):
    train_inds, test_inds = next(
        GroupShuffleSplit(test_size=test_ratio, random_state=random_state).split(df, groups=df['Patient ID']))

    return train_inds, test_inds, df.iloc[train_inds], df.iloc[test_inds]


def split_test_train_v3(df, splits_nr, test_ratio=0.2, random_state=None):
    '''
        :param df:
        :param splits_nr: the number of fold validation to make
        :param test_ratio:
        :param random_state: seed
        :return: If the splits_nr = 1: then it returns two arrays - of training indices and test indices
                If splits_nr > 1: the it return two lists. Each element is an arrays with training/test indices
                for each fold of the data
    '''
    # shuffle split ensuring that same patient ID is only in test or train
    gss = GroupShuffleSplit(n_splits=splits_nr, test_size=test_ratio, random_state=random_state)
    splits_iter = gss.split(df, groups=df['Patient ID'])

    if splits_nr == 1:
        train_inds, test_inds = next(splits_iter)
        return train_inds, test_inds, df.iloc[train_inds], df.iloc[test_inds]
    else:
        train_ind_col = []
        test_ind_col = []
        for split in splits_iter:
            # print((split))
            train_inds, test_inds = split
            train_ind_col.append(train_inds)
            test_ind_col.append(test_inds)
        # print(type(train_ind_col))
        return train_ind_col, test_ind_col
        # return train_inds, test_inds, df.iloc[train_inds], df.iloc[test_inds]


def split_test_train_stratified(df, test_ration, random_state = None):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)


def separate_localization_classification_labels(Y, single_class=None):
    if single_class is None:
        return Y.loc[Y['Bbox']==0], Y.loc[Y['Bbox']==1]
    else:
        class_ind = Y[single_class + '_loc'].isin(
            [str(np.zeros((16, 16))).replace('\n', ''), str(np.ones((16, 16))).replace('\n', '')])

        return Y.loc[class_ind], Y.loc[class_ind==False]
        # return Y.loc[Y[single_class+'_loc']==0], Y.loc[Y[single_class+'_loc']==1]

# THIS METHOD IS USED FOR KERAS TESTING
def keep_index_and_diagnose_columns(Y):
    return Y[['Dir Path', 'Atelectasis_loc', 'Cardiomegaly_loc', 'Consolidation_loc', 'Edema_loc',
        'Effusion_loc', 'Emphysema_loc','Fibrosis_loc', 'Hernia_loc', 'Infiltration_loc', 'Mass_loc',
        'Nodule_loc', 'Pleural_Thickening_loc', 'Pneumonia_loc', 'Pneumothorax_loc']]


def check_bounding_box_present(Y, class_name):
    Y[class_name + '_loc'] == str(np.zeros((16, 16))).replace('\n', '')


def keep_index_and_1diagnose_columns(Y, y_column_name):
    return Y[['Dir Path', y_column_name]]


def keep_observations_of_positive_patients(Y, res_path, class_name):
        Y['keep_patient'] = -1
        # keep_patient_flag = df.apply(lambda x: get_patient_substring(x), axis=1)
        res = Y.groupby(['Patient ID'])
        for name, group in res:
            keep_patient_flag = np.max(np.asarray(group[class_name]))
            Y.loc[Y['Patient ID'] == name, 'keep_patient'] = keep_patient_flag

        Y2 = Y.loc[(Y['keep_patient'])==1]
        Y2.to_csv(res_path + "processed_"+ class_name + ".csv")
        return Y2


def keep_observations_with_label(Y, class_name):
    return Y.loc[Y['Finding Labels'].str.contains(class_name)]


def get_rows_from_indices(df, train_inds, test_inds):
    return df.iloc[train_inds], df.iloc[test_inds]


# Lastly, We use 80% annotated images and 50% unanno-tated images to train the model and evaluate
#  on the other 20% annotated images in each fold.
def get_train_test(Y, random_state=None, do_stats=False, res_path =None, label_col=None):
    classification, bbox = separate_localization_classification_labels(Y, label_col)

    _, _, df_class_train, df_class_test = split_test_train_v2(classification, test_ratio=0.2, random_state=random_state)
    train_bbox_idx, _, df_bbox_train, df_bbox_test = split_test_train_v2(bbox, test_ratio=0.2, random_state=random_state)
    print("BBO TRAIN train test")
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

    label_patches = label_col + '_loc'
    print("Population: ")
    print("Train dataset ")
    print("No Finding: "+ str(df_train.loc[df_train['Finding Labels'].str.contains('No Finding')].shape[0]))
    print(label_col+ ": " + str(df_train.loc[df_train['Finding Labels'].str.contains(label_col)].shape[0]))
    print("Validation dataset ")
    print("No Finding: " + str(df_val.loc[df_val['Finding Labels'].str.contains('No Finding')].shape[0]))
    print(label_col + ": " + str(df_val.loc[df_val['Finding Labels'].str.contains(label_col)].shape[0]))
    print("Test without bounding boxes dataset ")
    print("No Finding: " + str(df_class_test.loc[df_class_test['Finding Labels'].str.contains('No Finding')].shape[0]))
    print(label_col + ": " + str(df_class_test.loc[df_class_test['Finding Labels'].str.contains(label_col)].shape[0]))
    if label_col is not None:
        train_set, val_set = keep_index_and_1diagnose_columns(df_train, label_patches),\
                             keep_index_and_1diagnose_columns(df_val,  label_patches)
        bbox_test, class_test = keep_index_and_1diagnose_columns(df_bbox_test,  label_patches),\
                                keep_index_and_1diagnose_columns(df_class_test,  label_patches)
        bbox_train = keep_index_and_1diagnose_columns(df_bbox_train, label_patches)

    return train_idx, train_set, val_set, bbox_test, class_test, bbox_train


def get_train_test_v2(Y, random_state=None, do_stats=False, res_path =None, label_col=None):
    classification, bbox = separate_localization_classification_labels(Y, label_col)

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

    label_patches = label_col + '_loc'
    if label_col is not None:
        train_set, val_set = keep_index_and_1diagnose_columns(df_train, label_patches),\
                             keep_index_and_1diagnose_columns(df_val,  label_patches)
        bbox_test, class_test = keep_index_and_1diagnose_columns(df_bbox_test,  label_patches),\
                                keep_index_and_1diagnose_columns(df_class_test,  label_patches)
        bbox_train = keep_index_and_1diagnose_columns(df_bbox_train, label_patches)

    return train_idx, train_set, val_set, bbox_test, class_test, bbox_train


def calculate_observations_to_keep(train_class_size, bbox_train_size, ratio_keep):
    '''
    :param train_clas_idx:  set of all train indices from which we will chose a ratio to keep
    :param bbox_train_size: bbox observations for training, they are separately from train_clas_idx
    :param ratio_keep: a ration between 0 and 1
    :return: computing total number of observations such we know how many observations to keep,
    and how many from CLASSIFICATION images we can keep, as we DO NOT drop any images with segmentation
    '''
    # print("class train indices are: " + str(len(train_clas_idx)))
    # print("Bbox train indices are: " + str(bbox_train_size))

    total_train_obs = train_class_size + bbox_train_size
    # print("total train opbservations" + str(total_train_obs))

    # TOTAL OBSERVATIONS OF THE NEW TRAIN SET
    obs_to_keep = np.ceil(ratio_keep * total_train_obs)
    # print("observations to keep" + str(obs_to_keep))

    # TOTAL observations which we need to draw from the classification train set
    obs_from_class_train = obs_to_keep - bbox_train_size
    # print("observation to keep in "+ str(obs_from_class_train))
    return obs_from_class_train


def construct_train_test_CV(df_class, class_train_col, class_test_col,
                            df_bbox, bbox_train_col, bbox_test_col,
                            val_ratio,
                            split, random_state, diagnose_col, train_subset_ratio=None):
    # print("classification")
    # print(class_train_col[splits_nr])
    # print(class_test_col[splits_nr])
    df_class_train, df_class_test = get_rows_from_indices(df_class, class_train_col[split],
                                                          class_test_col[split])
    df_bbox_train, df_bbox_test = get_rows_from_indices(df_bbox, bbox_train_col[split], bbox_test_col[split])

    print("BBOx TRAIN vs ")
    print(df_bbox_train.shape)
    print(df_bbox_test.shape)
    #     TODO: RENAMED DF_CLASS+TRAIN TO OLD_DF_CLASS_TRAIN
    train_clas_idx, _, complete_df_class_train, df_class_val = split_test_train_v3(df_class_train, 1, test_ratio=val_ratio,
                                                                          random_state=random_state)
    df_train = pd.concat([complete_df_class_train, df_bbox_train])


    df_val = df_class_val
    df_test = pd.concat([df_class_test, df_bbox_test])

    train_set, val_set = keep_index_and_1diagnose_columns(df_train, diagnose_col), \
                         keep_index_and_1diagnose_columns(df_val, diagnose_col)
    test_set = keep_index_and_1diagnose_columns(df_test, diagnose_col)
    bbox_train  = keep_index_and_1diagnose_columns(df_bbox_train, diagnose_col)
    bbox_test  = keep_index_and_1diagnose_columns(df_bbox_test, diagnose_col)
    complete_class_train = keep_index_and_1diagnose_columns(complete_df_class_train, diagnose_col)
    return train_set, val_set, test_set, bbox_train, bbox_test, complete_class_train


# Lastly, We use 80% annotated images and 50% unanno-tated images to train the model and evaluate
#  on the other 20% annotated images in each fold.
def get_train_test_CV(Y, splits_nr, current_split, random_seed,  label_col, ratio_to_keep=None):
    '''
    Returns a train-test separation in cross validation settings, for a specific split #
    If ratio_to_keep is specified, then (1-ratio) observations from the TRAIN set are dropped
    :param Y:
    :param splits_nr: total folds
    :param split: current fold
    :param random_state:
    :param label_col:
    :param ratio_to_keep: If None
    :return:
    '''
    classification, bbox = separate_localization_classification_labels(Y,  label_col)
    class_train_col, class_test_col = split_test_train_v3(classification, splits_nr=splits_nr, test_ratio=0.2,
                                                          random_state=random_seed)
    bbox_train_col, bbox_test_col = split_test_train_v3(bbox, splits_nr, test_ratio=0.2,
                                                        random_state=random_seed)
    # Here also train test is divided into train and validation
    train_set, val_set, test_set, df_bbox_train, df_bbox_test, train_only_class = construct_train_test_CV(classification, class_train_col, class_test_col,
                                                                    bbox, bbox_train_col, bbox_test_col,
                                                                    random_state=random_seed,
                                                                    diagnose_col=label_col+'_loc', split= current_split,
                                                                    val_ratio=0.2,
                                                                    train_subset_ratio=ratio_to_keep)
    print("bbox for train "+ str(df_bbox_train.shape))
    print("bbox for test "+ str(df_bbox_test.shape))
    print("total train set size is "+str(train_set.shape))
    print("önly classification set "+ str(train_only_class.shape))
    return train_set, val_set, test_set, df_bbox_train, df_bbox_test, train_only_class


def get_train_subset_xray(orig_train_set, train_bbox_nr, random_seed, ratio_to_keep):

    obs_to_keep = calculate_observations_to_keep(orig_train_set.shape[0], train_bbox_nr, ratio_to_keep)

    if obs_to_keep > 0:
        np.random.seed(seed=random_seed)
        class_train_ind_keep = np.random.choice(orig_train_set.index, int(obs_to_keep), replace=False)
        print("class train to keep")
        print(class_train_ind_keep)
        train_subset = orig_train_set.loc[class_train_ind_keep]
        return train_subset
    else:
        return orig_train_set


def load_xray(skip_processing, processed_labels_path, classication_labels_path, image_path, localization_labels_path,
              results_path, class_name):
    if skip_processing:
        xray_df = load_csv(processed_labels_path)
        print('Cardiomegaly label division')
        no_findings_samples = keep_observations_with_label(xray_df, "No Finding")[:10000]
        class_positive_samples = keep_observations_with_label(xray_df, class_name)
        filtered_patients_df = pd.concat([no_findings_samples, class_positive_samples])
        print(filtered_patients_df[class_name].value_counts())
    else:
        label_df = get_classification_labels(classication_labels_path, False)
        processed_df = preprocess_labels(label_df, image_path)
        xray_df = couple_location_labels(localization_labels_path, processed_df, PATCH_SIZE, results_path)
        # filtered_patients_df = keep_observations_of_positive_patients(xray_df, results_path, class_name)
        no_findings_samples = keep_observations_with_label(xray_df, "No Finding")
        class_positive_samples = keep_observations_with_label(xray_df, class_name)
        filtered_patients_df = pd.concat([no_findings_samples, class_positive_samples])
    return filtered_patients_df


def split_xray_cv(xray_df, cv_splits, split, class_name):
    df_train, df_val, df_test, \
    df_bbox_train, df_bbox_test, train_only_class = get_train_test_CV(xray_df, cv_splits, split, random_seed=1,
                                                                         label_col=class_name, ratio_to_keep=None)

    print('Training set: ' + str(df_train.shape))
    print('Validation set: ' + str(df_val.shape))
    print('Localization testing set: ' + str(df_test.shape))

    return df_train, df_val, df_test, df_bbox_train, df_bbox_test, train_only_class