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
    bbox = pd.read_csv(file_path)
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


def get_classification_labels(label_dir="C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv", preprocessed_csv=False):
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

    xy_df = xy_df.dropna(subset=['Dir Path'])
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


def keep_only_diagnose_columns(Y):
    return Y[['Atelectasis_loc', 'Cardiomegaly_loc', 'Consolidation_loc', 'Edema_loc',
        'Effusion_loc', 'Emphysema_loc','Fibrosis_loc', 'Hernia_loc', 'Infiltration_loc', 'Mass_loc',
        'Nodule_loc', 'Pleural_Thickening_loc', 'Pneumonia_loc', 'Pneumothorax_loc']]


# THIS METHOD IS USED FOR KERAS TESTING
def keep_index_and_diagnose_columns(Y):
    return Y[['Dir Path', 'Atelectasis_loc', 'Cardiomegaly_loc', 'Consolidation_loc', 'Edema_loc',
        'Effusion_loc', 'Emphysema_loc','Fibrosis_loc', 'Hernia_loc', 'Infiltration_loc', 'Mass_loc',
        'Nodule_loc', 'Pleural_Thickening_loc', 'Pneumonia_loc', 'Pneumothorax_loc']]


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
    # train_bbox_idx, _, df_bbox_train, df_bbox_val = split_test_train_v2(df_bbox_train, test_ratio=0.2, random_state=random_state)

    train_idx = np.concatenate((train_clas_idx, train_bbox_idx), axis=None)
    df_train = pd.concat([df_class_train, df_bbox_train])
    df_val = df_class_val
    # df_train= df_train.reindex(np.random.permutation(df_train.index))
    # df_val = df_val.reindex(np.random.permutation(df_val.index))

    if do_stats and res_path is not None:
        visualize_population(Y, 'whole_df_group', res_path, FINDINGS)
        visualize_population(df_train, 'train_group', res_path, FINDINGS)
        visualize_population(df_val, 'validation_group', res_path, FINDINGS)
        visualize_population(df_bbox_test, 'test_bbox_group', res_path, FINDINGS)
        visualize_population(df_class_test, 'test_class_group', res_path, FINDINGS)
        visualize_population(pd.concat([df_bbox_test, df_class_test]), 'test_group', res_path, FINDINGS)

    train_set, val_set = keep_index_and_diagnose_columns(df_train), keep_index_and_diagnose_columns(df_val)
    bbox_test, class_test = keep_index_and_diagnose_columns(df_bbox_test), keep_index_and_diagnose_columns(df_class_test)
    return train_idx, train_set, val_set, bbox_test, class_test


def sample_patients(max_observations_drop, df):
    patient_unq_indices = df['Patient ID'].unique()
    print("patient unique ind")
    print(patient_unq_indices)


    np.random.seed(2)
    max_drop_patients_ind = np.random.choice(patient_unq_indices, max_observations_drop,
                                         replace=False)

    nr_obs_dropped = 0
    last_pat_ind = 0

    for i in range(0, len(max_drop_patients_ind)):
        nr_obs_dropped += df.loc[df['Patient ID'] == max_drop_patients_ind[i]].shape[0]
        last_pat_ind = i
        ratio = nr_obs_dropped/max_observations_drop
        if 1.1 <= ratio <= 0.9:
            max_drop_patients_ind = max_drop_patients_ind[0:i + 1]

    return max_drop_patients_ind, nr_obs_dropped, max_observations_drop


def create_overlap_testing_v2(min_overlap, start_seed, init_df, enrich_df):
    # pat_drop_ub = np.math.floor((1 - min_overlap) * len(init_df['Patient ID'].unique()))
    max_obs_drop = np.math.floor((1 - min_overlap) * init_df.shape[0])

    drop_patients_ind, nr_obs_dropped, max_obs_todrop= sample_patients(max_obs_drop, init_df)

    # print(df.loc[df['Patient ID']==pat_ind] for pat_ind in drop_patients_ind)

    nr_obs_dropped = np.sum([init_df.loc[init_df['Patient ID']==pat_ind].shape[0] for pat_ind in drop_patients_ind])
    print("patients to drop")
    print(drop_patients_ind)
    print(nr_obs_dropped)
    # add_patients_ind, nr_obs_added, max_pat_todrop= sample_patients(min_overlap, init_df)

    add_patients_ind, nr_obs_added, max_obs_toadd = sample_patients(nr_obs_dropped, enrich_df)
    #
    # patient_unq_indices_enrich = enrich_df['Patient ID'].unique()
    # print(patient_unq_indices_enrich)
    # np.random.seed(1)
    # new_patients_ind = np.random.choice(patient_unq_indices_enrich, nr_obs_added, replace=False) # for _ in range(total_drop_patients)]
    # print("new patients id")
    # print(new_patients_ind)
    # nr_obs_added = 0
    #
    # for pat_ind in new_patients_ind:
    #     print("for loop 2")
    #     print(enrich_df.loc[enrich_df['Patient ID'] == pat_ind].shape[0])
    #     nr_obs_added += enrich_df.loc[enrich_df['Patient ID'] == pat_ind].shape[0]
    #     if (nr_obs_added >= nr_obs_dropped):
    #         ratio_init_df = (init_df.shape[0]-nr_obs_dropped)/init_df.shape[0]
    #         ratio_enriched_df = (init_df.shape[0]-nr_obs_dropped)/(init_df.shape[0]-nr_obs_dropped + nr_obs_added)
    #         print("ratios")
    #         print(ratio_init_df)
    #         print(ratio_enriched_df)
    #         # break
    #     print("added")
    # print(nr_obs_added)



def create_overlapping_test_set(init_train_idx, start_seed, max_overlap, min_overlap, df):
    print("train indices")
    print(init_train_idx)
    seed = start_seed
    new_train_idx = []
    overlap = np.intersect1d(init_train_idx,new_train_idx)
    overlap_ratio = float(len(overlap)) / len(init_train_idx)
    print("overlap ratio is:")
    print(overlap_ratio)
    while(not(max_overlap > overlap_ratio and overlap_ratio>min_overlap)):
        seed+=1
        _, df_train, df_val, df_bbox_test, df_class_test = get_train_test(df, random_state=seed, do_stats=False, res_path=None)
        new_train_idx = df_train['Dir Path'].index.values
        overlap = np.intersect1d(init_train_idx, new_train_idx)
        overlap_ratio = float(len(overlap)) / len(init_train_idx)
        print("overlap ratio is:")
        print(overlap_ratio)
        print(seed)
        if (seed== (start_seed+1)*1000):
            print("No overlapping training test can be constructed within 10000 iterations")
            break
    return seed, new_train_idx


def create_overlapping_set(init_train_idx, start_seed, max_overlap, min_overlap, df):
    print("train indices")
    print(init_train_idx)
    seed = start_seed
    new_train_idx = []
    overlap = np.intersect1d(init_train_idx,new_train_idx)
    overlap_ratio = float(len(overlap)) / len(init_train_idx)
    while(not(max_overlap > overlap_ratio and overlap_ratio>min_overlap)):
        seed+=1
        new_train_idx, _, _, _, _ = get_train_test(df, random_state=seed, do_stats=False, res_path=None)
        overlap = np.intersect1d(init_train_idx, new_train_idx)
        overlap_ratio = float(len(overlap)) / len(init_train_idx)
        print(seed)
        if (seed== (start_seed+1)*1000):
            print("No overlapping training test can be constructed within 10000 iterations")
            break
    return seed, new_train_idx


def keep_only_classification_columns(Y):
    return Y[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
              'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
              'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']]


def reorder_Y(Y):
    newarr = []
    for i in range(0, Y.shape[0]):  # (14)
        single_obs = []
        for j in range(Y.shape[1]):  # 1 -> (16, 16)
            single_obs.append(Y.iloc[i,j])
        newarr.append(single_obs)
    return np.transpose(np.asarray(newarr), [0, 2, 3, 1])


#################### processing loaded data ######################


def select_y_class_columns(df):
    return df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                      'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                      'Pneumonia', 'Pneumothorax']].astype(np.int64)


def multilabel_stratification(df, Y, splitnr, rnd_seed = 0):
    mskf = MultilabelStratifiedKFold(n_splits=splitnr, random_state=rnd_seed)

    for train_index, test_index in mskf.split(np.zeros(Y.shape[0]), Y.values):
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        # y_train, y_test = Y[train_index], Y[test_index]
        return train_index, test_index, df_train, df_test


def train_test_stratification(xray_df, rnd_seed=0):
    classification, bbox = separate_localization_classification_labels(xray_df)
    y_class = select_y_class_columns(classification)
    y_bbox = select_y_class_columns(bbox)
    # print("testing stratification")
    # print("classif train val - test")
    ### 50% unannotated images to train model
    _, _, df_class_trainval, df_class_test = multilabel_stratification(classification, y_class, 2, rnd_seed=rnd_seed)

    ### 80% annotated images for training
    _, _, df_bbox_trainval, df_bbox_test = multilabel_stratification(bbox, y_bbox, 5, rnd_seed=rnd_seed)
    # print("classif train -val ")

    ### 20% of train data is validation - split = 5
    _, _, df_class_train, df_class_val = multilabel_stratification(df_class_trainval, select_y_class_columns(df_class_trainval), 5, rnd_seed=rnd_seed)

    ## 20% of train data is validation - split = 5
    _, _, df_bbox_train, df_bbox_val = multilabel_stratification(df_bbox_trainval, select_y_class_columns(df_bbox_trainval), 5, rnd_seed=rnd_seed)

    df_train = pd.concat([df_class_train, df_bbox_train])
    df_val = pd.concat([df_class_val, df_bbox_val])


    return df_train.reindex(np.random.permutation(df_train.index)), df_val.reindex(np.random.permutation(df_val.index)), \
           df_class_test, df_bbox_test


def get_train_test_strata(xray_df, random_state=0, do_stats=True, res_path= None):
    df_train, df_val, df_class_test, df_bbox_test = train_test_stratification(xray_df, rnd_seed=random_state)

    if do_stats and res_path is not None:
        visualize_population(xray_df, 'whole_df_strata', res_path, FINDINGS)
        visualize_population(df_train, 'train_strata', res_path, FINDINGS)
        visualize_population(df_val, 'validation_strata', res_path, FINDINGS)
        visualize_population(df_bbox_test, 'test_bbox_strata', res_path, FINDINGS)
        visualize_population(df_class_test, 'test_class_strata', res_path, FINDINGS)

    train_set, val_set = keep_index_and_diagnose_columns(df_train), keep_index_and_diagnose_columns(df_val)
    bbox_test, class_test = keep_index_and_diagnose_columns(df_bbox_test), keep_index_and_diagnose_columns(
        df_class_test)
    return train_set, val_set, bbox_test, class_test