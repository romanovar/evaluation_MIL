import pandas as pd
import numpy as np
from keras.preprocessing import image
import os
from pathlib import Path
from keras.applications.resnet50 import preprocess_input
import cv2
from sklearn.model_selection import GroupShuffleSplit

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


def load_image(img_path):
        png = cv2.imread(img_path, 0)


def reorder_rows(df):
    return df.sort_values(by=["Reorder Index"])


def add_reorder_indx(df, image_ind, reord_idx):
    idx  = (df['Image Index']== image_ind)

    #IF image found in the label file
    if (df.loc[idx]).empty == False:
        df.loc[idx, 'Image Found'] = 1
        df.loc[idx, 'Reorder Index'] = reord_idx
        reord_idx+=1
    return df, reord_idx


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


def load_process_png_v2(Yclass, path_to_png):
    xy_df = Yclass.copy(deep=True)
    xy_df['Image Found'] = None
    xy_df['Reorder Index'] = None
    xy_df['Dir Path'] = None

    png_files = []
    reord_ind = 0
    for src_path in Path(path_to_png).glob('**/*.png'):

        image_ind = os.path.basename(src_path)

        xy_df.loc[xy_df['Image Index'] == image_ind, ['Dir Path']] = str(src_path)

        # img_data = process_image(src_path)
        xy_df, reord_ind = add_reorder_indx(xy_df, image_ind, reord_idx=reord_ind)

        png_files.append(process_image(src_path))
        # todo: DECIDE WHETHER TO DROP IMAGE ID here or later
        # drop column with img ind in Y
        # Y = Y.iloc[:, 2:Y.shape[1]]
        # labels.append(np.array(Y.values))
    # TODO: uncomment the dropping procedure
    xy_df = xy_df.dropna(subset=['Image Found'])
    return np.array(png_files), reorder_rows(xy_df) #def drop_extra_label_columns(xy_df)


def translate_on_patches(x_min, y_min, x_max, y_max):
    x = int(np.round((x_min/IMAGE_X)*16))
    y = int(np.round((y_min/IMAGE_Y)*16))
    x_max = int(np.round((x_max/IMAGE_X)*16))
    y_max = int(np.round((y_max/IMAGE_Y)*16))
    return x, y, x_max, y_max


def build_label_matrices(Y_):
    return 0


def bind_location_labels(Y_loc_dir, Y_class, P):
    Y_loc = load_csv(Y_loc_dir)
    Y_bbox = rename_columns(Y_loc, False)
    for ind, row in Y_bbox.iterrows():
        Y_class.loc[Y_class['Image Index']== row['Image Index'],'Bbox']=1
        Y_class.loc[Y_class['Image Index']== row['Image Index'],row['Finding Label']+'_loc']=1

        src_path= (Y_class.loc[Y_class['Image Index'] == row['Image Index'], 'Dir Path']).values

        if not src_path.size==0:
            scale_x, scale_y = scaling_factor(src_path[0])
            x_min, y_min, x_max, y_max = translate_coords_to_new_image_size(row['x'], row['y'], row['w'], row['h'], scale_x, scale_y)

            x_min, y_min, x_max, y_max = translate_on_patches(x_min, y_min, x_max, y_max )

            im_q = np.zeros((P, P), np.float)
            im_q = cv2.rectangle(im_q, (x_min, y_min), (x_max, y_max), 1, -1)

    Y_class.to_csv("C:/Users/s161590/Desktop/Data/X_Ray/processed_Y.csv")
    return Y_class, Y_loc


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


def make_label_matrix_localization(P, x_min, y_min, x_max, y_max):
    im_q = np.zeros((P, P), np.float)
    im_q = cv2.rectangle(im_q, (x_min, y_min), (x_max, y_max), 1, -1)
    return im_q


def get_all_bbox_for_image(row, Y_loc):
    all_info = Y_loc.loc[Y_loc['Image Index'] == row['Image Index']]
    return all_info, row


def integrate_annotations(row, Y_loc, diagnosis, P):
    result_image_class = []
    all_rows, row_classif_df = get_all_bbox_for_image(row, Y_loc)

    # if no bbox is found for this imae
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
            scale_x, scale_y = scaling_factor(row_classif_df['Dir Path'])
            x_min, y_min, x_max, y_max = translate_coords_to_new_image_size(row['x'], row['y'], row['w'], row['h'],
                                                                            scale_x,
                                                                            scale_y)
            x_min, y_min, x_max, y_max = translate_on_patches(x_min, y_min, x_max, y_max)
            y_mat = make_label_matrix_localization(PATCH_SIZE, x_min, y_min, x_max, y_max)
            return y_mat
    else:
        print("this hsould NOT BE PRINTING ")




def image_with_bbox(row, Y_loc, diagnosis, P):
    bbox_diagnosis = Y_loc.loc[Y_loc['Image Index'] == row['Image Index'], 'Finding Label'].values

    if bbox_diagnosis.size==0:
        # do SMTH WHEN THE IMAGE DOES NOT HAVE ANY BBOX FOR THE CLASS
        # CHECK IF CLASS IS 1
        y_mat = create_label_matrix_classification(row, diagnosis, P)
        # return row['Image Index'], diagnosis, y_mat
        return y_mat
    # IS THERE A BBOX FOR THIS IMAGE
    if not bbox_diagnosis.size == 0:
        if diagnosis in bbox_diagnosis:
            for diag in bbox_diagnosis:
                if diagnosis == diag:
                    src_path = row['Dir Path']
                    scale_x, scale_y = scaling_factor(src_path)
                    print("looking for astring")
                    print(row['Image Index'])
                    # print(scale_x)
                    print(row['x'], row['y'], row['w'], row['h'])
                    return row['x']
                    # x_min, y_min, x_max, y_max = translate_coords_to_image_size(row['x'], row['y'], row['w'], row['h'],
                    #                                                             scale_x,
                    #                                                             scale_y)
                    # x_min, y_min, x_max, y_max = translate_on_patches(x_min, y_min, x_max, y_max)

                    # y_mat = make_label_matrix_localization(PATCH_SIZE, x_min, y_min, x_max, y_max)
                    # return y_mat
        else:
            y_mat = create_label_matrix_classification(row, diagnosis, P)
            return y_mat
    else:
        return 0
            #     else:
        #         return ("fix")
    #     for label in FINDINGS:
    #         print("hdksjhdkaiofdaj")
    #     #     print(label)
    #         # print(fl)
    #         print(label in fl)
            # for bbox_class in range(fl.size):
            #     pass


def translate_coords(row, x, y, w, h, scale_x, scale_y, img_size):
    translate_coords_to_new_image_size(x, y, w, h, scale_x, scale_y)


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


def split_test_train_v2(df, test_ratio=0.2):
    # shuffle split ensuring that same patient ID is only in test or train
    train_inds, test_inds = next(GroupShuffleSplit(test_size=test_ratio, random_state=0).split(df, groups=df['Patient ID']))

    return df.iloc[train_inds], df.iloc[test_inds]


def separate_localization_classification_labels(Y):
    return Y.loc[Y['Bbox']==0], Y.loc[Y['Bbox']==1]


def keep_only_diagnose_columns(Y):
    return Y[['Atelectasis_loc', 'Cardiomegaly_loc', 'Consolidation_loc', 'Edema_loc',
        'Effusion_loc', 'Emphysema_loc','Fibrosis_loc', 'Hernia_loc', 'Infiltration_loc', 'Mass_loc',
        'Nodule_loc', 'Pleural_Thickening_loc', 'Pneumonia_loc', 'Pneumothorax_loc']]


# THIS METHOD IS USED FOR KERAS TESTING
def keep_index_and_diagnose_columns(Y):
    return Y[['Image Index', 'Atelectasis_loc', 'Cardiomegaly_loc', 'Consolidation_loc', 'Edema_loc',
        'Effusion_loc', 'Emphysema_loc','Fibrosis_loc', 'Hernia_loc', 'Infiltration_loc', 'Mass_loc',
        'Nodule_loc', 'Pleural_Thickening_loc', 'Pneumonia_loc', 'Pneumothorax_loc']]


# Lastly, We use 80% annotated images and 50% unanno-tated images to train the model and evaluate
#  on the other 20% annotated images in each fold.
def get_train_test(Y):
    classification, bbox = separate_localization_classification_labels(Y)
    # clas, local
    # classification, bbox = keep_index_and_diagnose_columns(clas), keep_index_and_diagnose_columns(local)

    df_class_train, df_class_test = split_test_train_v2(classification, test_ratio=0.5)
    df_bbox_train, df_bbox_test = split_test_train_v2(bbox, test_ratio=0.8)

    df_class_train, df_class_val = split_test_train_v2(df_class_train, test_ratio=0.2)
    df_bbox_train, df_bbox_val = split_test_train_v2(df_bbox_train, test_ratio=0.2)

    df_train = pd.concat([df_class_train, df_bbox_train])
    df_val = pd.concat([df_class_val, df_bbox_val])
    # print("classification train set: ")
    # print(df_class_train['Bbox'].value_counts())
    # print("localization train set: ")
    # print(df_bbox_train['Bbox'].value_counts())
    #
    # print("classification validation set: ")
    # print(df_class_val['Bbox'].value_counts())
    # print("localization validation set: ")
    # print(df_bbox_val['Bbox'].value_counts())
    #
    # print("test test set: ")
    # print(df_class_test['Bbox'].value_counts())
    # print("localization test set: ")
    # print(df_bbox_test['Bbox'].value_counts())

    train_set, val_set = keep_index_and_diagnose_columns(df_train), keep_index_and_diagnose_columns(df_val)
    bbox_test, class_test = keep_index_and_diagnose_columns(df_bbox_test), keep_index_and_diagnose_columns(df_class_test)
    return train_set, val_set, bbox_test, class_test

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
