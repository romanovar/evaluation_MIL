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

from keras_utils import plot_grouped_bar_population, plot_pie_population, visualize_population

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
# TODO: remove if not used
def get_classification_labels(label_dir="C:/Users/s161590/Desktop/Data/X_Ray/Data_Entry_2017.csv", preprocessed_csv=False):
    Y = load_csv(label_dir)
    # TODO: turn into new preparation function
    if not preprocessed_csv:
        Y_pr = rename_columns(Y, True)
        Y = add_label_columns(Y_pr)
    # SAVE CSV
    #     Y.to_csv("C:/Users/s161590/Desktop/Data/X_Ray/processed_Y.csv")
    # Y1 = Y[['Image Index', 'Patient ID', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
    #         'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
    #         'Pneumonia', 'Pneumothorax']]
    return Y



def drop_extra_label_columns(df):
    # dropping 'Image Index', 'Patient ID', Image Found and ReorderIndex
    return df[[ 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
         'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
         'Pneumonia', 'Pneumothorax']]


def get_input_df(df):
    return df [['Image Index', 'Image Found']]

def process_image(img_path):
    img = image.load_img(img_path, target_size=(512, 512))
    x = image.img_to_array(img)
    return preprocess_input(x)


def scaling_factor(img_path):
    img = cv2.imread(img_path, 0)
    x, y = img.shape[0], img.shape[1]
    return x/IMAGE_X, y/IMAGE_Y


def scaling_factor_v2(img_path):
    x, y = imagesize.get(img_path)

    return x/IMAGE_X, y/IMAGE_Y

def load_image(img_path):
        png = cv2.imread(img_path, 0)


def reorder_rows(df):
    return df.sort_values(by=["Reorder Index"])


# def add_reorder_indx(df, image_ind, reord_idx):
#     idx  = (df['Image Index']== image_ind)
#
#     #IF image found in the label file
#     if (df.loc[idx]).empty == False:
#         df.loc[idx, 'Image Found'] = 1
#         df.loc[idx, 'Reorder Index'] = reord_idx
#         reord_idx+=1
#     return df, reord_idx


def load_process_png(label_df, path_to_png):
    """
    Searches recursively for png files starting from the common parent dir
    When png files is found, it is loaded and added to the final list
    :param path_to_png: common parent directory path for all dicom files
    :return: list with all loaded dicom files found
    """
    png_files = []
    labels = []
    labels_reordered = pd.DataFrame()  # creates a new dataframe that's empty
    for src_path in Path(path_to_png).glob('**/*.png'):
        image_ind = os.path.basename(src_path)
        Y = get_label_by_imageind(label_df, image_ind)
        png_files.append(process_image(src_path))
        # todo: DECIDE WHETHER TO DROP IMAGE ID here or later
        # drop column with img ind in Y
        Y = Y.iloc[:, 2:Y.shape[1]]
        labels.append(np.array(Y.values))
    return np.array(png_files), np.array(labels)


# def load_process_png_v2(Yclass, path_to_png):
#     xy_df = Yclass.copy(deep=True)
#     xy_df['Image Found'] = None
#     xy_df['Reorder Index'] = None
#     xy_df['Dir Path'] = None
#
#     png_files = []
#     reord_ind = 0
#     for src_path in Path(path_to_png).glob('**/*.png'):
#
#         image_ind = os.path.basename(src_path)
#
#         xy_df.loc[xy_df['Image Index'] == image_ind, ['Dir Path']] = str(src_path)
#
#         # img_data = process_image(src_path)
#         xy_df, reord_ind = add_reorder_indx(xy_df, image_ind, reord_idx=reord_ind)
#
#         png_files.append(process_image(src_path))
#         # todo: DECIDE WHETHER TO DROP IMAGE ID here or later
#         # drop column with img ind in Y
#         # Y = Y.iloc[:, 2:Y.shape[1]]
#         # labels.append(np.array(Y.values))
#     # TODO: uncomment the dropping procedure
#     xy_df = xy_df.dropna(subset=['Image Found'])
#     return np.array(png_files), reorder_rows(xy_df) #def drop_extra_label_columns(xy_df)


def preprocess_labels(Yclass, path_to_png):
    xy_df = Yclass.copy(deep=True)
    xy_df['Image Found'] = None
    xy_df['Reorder Index'] = None
    xy_df['Dir Path'] = None

    # png_files = []
    reord_ind = 0
    for src_path in Path(path_to_png).glob('**/*.png'):

        image_ind = os.path.basename(src_path)

        xy_df.loc[xy_df['Image Index'] == image_ind, ['Dir Path']] = str(src_path)

        # img_data = process_image(src_path)
        # xy_df, reord_ind = add_reorder_indx(xy_df, image_ind, reord_idx=reord_ind)

        # png_files.append(process_image(src_path))
        # todo: DECIDE WHETHER TO DROP IMAGE ID here or later
        # drop column with img ind in Y
        # Y = Y.iloc[:, 2:Y.shape[1]]
        # labels.append(np.array(Y.values))
    # TODO: uncomment the dropping procedure
    # xy_df = xy_df.dropna(subset=['Image Found'])
    xy_df = xy_df.dropna(subset=['Dir Path'])
    return reorder_rows(xy_df) #def drop_extra_label_columns(xy_df)


def translate_on_patches(x_min, y_min, x_max, y_max):
    x = int(np.round((x_min/IMAGE_X)*PATCH_SIZE))
    y = int(np.round((y_min/IMAGE_Y)*PATCH_SIZE))
    x_max = int(np.round((x_max/IMAGE_X)*PATCH_SIZE))
    y_max = int(np.round((y_max/IMAGE_Y)*PATCH_SIZE))
    return x, y, x_max, y_max


#
# def bind_location_labels(Y_loc_dir, Y_class, P):
#     Y_loc = load_csv(Y_loc_dir)
#     Y_bbox = rename_columns(Y_loc, False)
#     for ind, row in Y_bbox.iterrows():
#         Y_class.loc[Y_class['Image Index']== row['Image Index'],'Bbox']=1
#         Y_class.loc[Y_class['Image Index']== row['Image Index'],row['Finding Label']+'_loc']=1
#
#         src_path= (Y_class.loc[Y_class['Image Index'] == row['Image Index'], 'Dir Path']).values
#
#         if not src_path.size==0:
#             scale_x, scale_y = scaling_factor(src_path[0])
#             x_min, y_min, x_max, y_max = translate_coords_to_new_image_size(row['x'], row['y'], row['w'], row['h'], scale_x, scale_y)
#
#             x_min, y_min, x_max, y_max = translate_on_patches(x_min, y_min, x_max, y_max )
#
#             im_q = np.zeros((P, P), np.float)
#             im_q = cv2.rectangle(im_q, (x_min, y_min), (x_max, y_max), 1, -1)
#
#     Y_class.to_csv("C:/Users/s161590/Desktop/Data/X_Ray/processed_Y.csv")
#     return Y_class, Y_loc


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


# def make_label_matrix_localization(P, x_min, y_min, x_max, y_max):
#     im_q = np.zeros((P, P), np.float)
#     im_q = cv2.rectangle(im_q, (x_min, y_min), (x_max, y_max), 1, -1)
#     print("make label matrix localization ")
#     print(x_min)
#     print(x_max)
#     print(y_min)
#     print(y_max)
#     print(im_q)
#     return im_q

def make_label_matrix_localization_v2(P, x_min, y_min, x_max, y_max):
    im_q = np.zeros((P, P), np.float)
    im_q[y_min:(y_max + 1), x_min:(x_max + 1)] = 1.
    # print(x_min)
    # print(x_max)
    # print(y_min)
    # print(y_max)
    # print(im_q)
    # im_q = cv2.rectangle(im_q, (x_min, y_min), (x_max, y_max), 1, -1)
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


#
#
# def image_with_bbox(row, Y_loc, diagnosis, P):
#     bbox_diagnosis = Y_loc.loc[Y_loc['Image Index'] == row['Image Index'], 'Finding Label'].values
#
#     if bbox_diagnosis.size==0:
#         # do SMTH WHEN THE IMAGE DOES NOT HAVE ANY BBOX FOR THE CLASS
#         # CHECK IF CLASS IS 1
#         y_mat = create_label_matrix_classification(row, diagnosis, P)
#         # return row['Image Index'], diagnosis, y_mat
#         return y_mat
#     # IS THERE A BBOX FOR THIS IMAGE
#     if not bbox_diagnosis.size == 0:
#         if diagnosis in bbox_diagnosis:
#             for diag in bbox_diagnosis:
#                 if diagnosis == diag:
#                     src_path = row['Dir Path']
#                     scale_x, scale_y = scaling_factor(src_path)
#                     print("looking for astring")
#                     print(row['Image Index'])
#                     # print(scale_x)
#                     print(row['x'], row['y'], row['w'], row['h'])
#                     return row['x']
#                     # x_min, y_min, x_max, y_max = translate_coords_to_image_size(row['x'], row['y'], row['w'], row['h'],
#                     #                                                             scale_x,
#                     #                                                             scale_y)
#                     # x_min, y_min, x_max, y_max = translate_on_patches(x_min, y_min, x_max, y_max)
#
#                     # y_mat = make_label_matrix_localization(PATCH_SIZE, x_min, y_min, x_max, y_max)
#                     # return y_mat
#         else:
#             y_mat = create_label_matrix_classification(row, diagnosis, P)
#             return y_mat
#     else:
#         return 0
#             #     else:
#         #         return ("fix")
#     #     for label in FINDINGS:
#     #         print("hdksjhdkaiofdaj")
#     #     #     print(label)
#     #         # print(fl)
#     #         print(label in fl)
#             # for bbox_class in range(fl.size):
#             #     pass


# def translate_coords(row, x, y, w, h, scale_x, scale_y, img_size):
#     translate_coords_to_new_image_size(x, y, w, h, scale_x, scale_y)
#

def translate_coords_to_new_image_size(x, y, w, h, scale_x, scale_y):
    x_min = x/ scale_x
    y_min = y / scale_y
    x_max = x_min + w / scale_x
    y_max = y_min + h / scale_y
    return x_min, y_min, x_max, y_max


def bbox_available(df, img_ind):
    return df.loc[df['Image Index'] == img_ind, ['Bbox']]


# ### LOOP through and update the bbox, according to the
# def get_bbox_coords(class_df, loc_df, img_ind):
#     # image in bbox coord may appear multiple times - 1 row per class
#     rows = loc_df.loc[loc_df['Image Index'] == img_ind]
#     # Images in classification file will appear once, while
#     for ind, row in loc_df.iterrows():
#         res.append([row['Finding Label'], row['x'], row['y'], row['w'], row['h']])
#         if bbox_available(Yclass, image_ind) == 1:
#             coords = get_bbox_coords()


# def add_bbox_coord():
#
#     coords = str(row['x']) + ', ' + str(row['y']) + ', ' + str(row['w']) + ', ' + str(row['h'])
#
#     # x y] are coordinates of each box's topleft corner. [w h] represent the width and height of each box
#     Y_class.loc[Y_class['Image Index'] == row['Image Index'], [row['Finding Label'] + '_coord']] = coords

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
# bbox = load_csv("C:/Users/s161590/Desktop/Data/X_Ray/BBox_List_2017.csv")
#
# image_ind_with_bbox = get_ann_list(bbox)
# # find_load_annotated_png_files(image_ind_with_bbox, path_to_png="C:/Users/s161590/Desktop/Data/X_Ray/images")
#

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
    # clas, local
    # classification, bbox = keep_index_and_diagnose_columns(clas), keep_index_and_diagnose_columns(local)

    _, _, df_class_train, df_class_test = split_test_train_v2(classification, test_ratio=0.5, random_state=random_state)
    _, _, df_bbox_train, df_bbox_test = split_test_train_v2(bbox, test_ratio=0.8, random_state=random_state)

    train_clas_idx, _, df_class_train, df_class_val = split_test_train_v2(df_class_train, test_ratio=0.2, random_state=random_state)
    train_bbox_idx, _, df_bbox_train, df_bbox_val = split_test_train_v2(df_bbox_train, test_ratio=0.2, random_state=random_state)

    train_idx = np.concatenate((train_clas_idx, train_bbox_idx), axis=None)
    df_train = pd.concat([df_class_train, df_bbox_train])
    df_val = pd.concat([df_class_val, df_bbox_val])
    df_train= df_train.reindex(np.random.permutation(df_train.index))
    df_val = df_val.reindex(np.random.permutation(df_val.index))

    if do_stats and res_path is not None:
        visualize_population(Y, 'whole_df_group', res_path, FINDINGS)
        visualize_population(df_train, 'train_group', res_path, FINDINGS)
        visualize_population(df_val, 'validation_group', res_path, FINDINGS)
        visualize_population(df_bbox_test, 'test_bbox_group', res_path, FINDINGS)
        visualize_population(df_class_test, 'test_class_group', res_path, FINDINGS)
        # plot_grouped_bar_population(Y, 'whole_df', res_path, FINDINGS)
        # plot_grouped_bar_population(df_train, 'train', res_path, FINDINGS)
        # plot_grouped_bar_population(df_val, 'validation', res_path, FINDINGS)
        # plot_grouped_bar_population(df_bbox_test, 'test_bbox', res_path, FINDINGS)
        # plot_grouped_bar_population(df_class_test, 'test_class', res_path, FINDINGS)
        # plot_pie_population(Y, 'whole_df', res_path, FINDINGS)
        # plot_pie_population(df_train, 'train', res_path, FINDINGS)
        # plot_pie_population(df_val, 'validation', res_path, FINDINGS)
        # plot_pie_population(df_bbox_test, 'test_bbox', res_path, FINDINGS)
        # plot_pie_population(df_class_test, 'test_class', res_path, FINDINGS)
    print(train_idx)
    train_set, val_set = keep_index_and_diagnose_columns(df_train), keep_index_and_diagnose_columns(df_val)
    bbox_test, class_test = keep_index_and_diagnose_columns(df_bbox_test), keep_index_and_diagnose_columns(df_class_test)
    return train_idx, train_set, val_set, bbox_test, class_test


def create_overlapping_test_set(init_train_idx, start_seed, max_overlap, min_overlap, df):
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

    # print("found")
    # print(seed)
    # print(overlap)
    # print(len(init_train_idx))
    # print(new_train_idx)
    # print((max_overlap > overlap_ratio))
    # print(overlap_ratio<min_overlap)
    # print(overlap_ratio)
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


def process_loaded_labels_tf(label_col):
    newstr = (label_col.replace("[", "")).replace("]", "")
    return np.fromstring(newstr, dtype=np.ones((PATCH_SIZE, PATCH_SIZE)).dtype, sep=' ').reshape(PATCH_SIZE, PATCH_SIZE)


def select_y_class_columns(df):
    return df[['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                      'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                      'Pneumonia', 'Pneumothorax']].astype(np.int64)



def multilabel_stratification(df, Y, splitnr, rnd_seed = 0):
    mskf = MultilabelStratifiedKFold(n_splits=splitnr, random_state=rnd_seed)
    X = np.zeros(Y.shape[0])
    # X.values[:, np.newaxis]
    for train_index, test_index in mskf.split(np.zeros(Y.shape[0]), Y.values):
        # print("TRAIN your set:", train_index, "TEST your set:", test_index)
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