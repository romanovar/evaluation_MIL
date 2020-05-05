import cnn.preprocessor.load_data as ld
import cnn.preprocessor.load_data_mura as ldm
import cnn.preprocessor.load_data_pascal as ldp
from cnn.keras_utils import  process_loaded_labels
import numpy as np


def test_size_bbox(bbox_label):
    array_numb = process_loaded_labels(bbox_label)
    sum_active_patches = np.sum(np.asarray(array_numb))
    return sum_active_patches


def load_process_xray14(config):
    skip_processing = config['skip_processing_labels']
    image_path = config['image_path']
    processed_labels_path = config['processed_labels_path']
    classication_labels_path = config['classication_labels_path']
    localization_labels_path = config['localization_labels_path']
    results_path = config['results_path']
    class_name = config['class_name']

    xray_df = ld.load_xray(skip_processing, processed_labels_path, classication_labels_path, image_path,
                           localization_labels_path, results_path)
    xray_df = ld.filter_observations(xray_df, class_name, 'No Finding')
    print(xray_df.shape)
    print("Splitting data ...")

    df_train, df_val, df_test = ld.get_train_test(xray_df, random_state=1, do_stats=False,
                                                                   res_path = results_path,
                                                                   label_col = class_name)

    label_patches = class_name + '_loc'
    if class_name is not None:
        df_train_filtered_cols, df_val_filtered_cols, df_test_filtered_cols =\
            ld.keep_index_and_1diagnose_columns(df_train, label_patches), \
            ld.keep_index_and_1diagnose_columns(df_val,  label_patches),\
            ld.keep_index_and_1diagnose_columns(df_test,  label_patches)
    return df_train_filtered_cols, df_val_filtered_cols, df_test_filtered_cols


def load_preprocess_mura(config):
    mura_train_img_path = config['mura_train_img_path']
    mura_train_labels_path = config['mura_train_labels_path']
    mura_test_img_path = config['mura_test_img_path']
    mura_test_labels_path = config['mura_test_labels_path']
    processed_train_labels_path = config['mura_processed_train_labels_path']
    processed_test_labels_path = config['mura_processed_test_labels_path']

    skip_processing = config['skip_processing_labels']

    class_name = config['class_name']
    ldm.check_validity_class(class_name)
    df_train_val, test_df_all_classes = ldm.load_mura(skip_processing, processed_train_labels_path,
                                                      processed_test_labels_path, mura_train_img_path,
                                                      mura_train_labels_path, mura_test_labels_path, mura_test_img_path)

    df_train_final, df_val_final, df_test_final= ldm.prepare_mura_set(df_train_val, test_df_all_classes, class_name)

    return df_train_final, df_val_final, df_test_final


def load_preprocess_pascal(config):
    pascal_image_path = config['pascal_image_path']
    df = ldp.load_pascal(pascal_image_path)
    return ldp.split_train_val_test(df)


#TODO: delete if not used

# def load_data_cv(config, data_set_name):
#     use_xray_dataset = config['use_xray_dataset']
#     skip_processing = config['skip_processing_labels']
#     image_path = config['image_path']
#     classication_labels_path = config['classication_labels_path']
#     localization_labels_path = config['localization_labels_path']
#     results_path = config['results_path']
#     processed_labels_path = config['processed_labels_path']
#
#
#     if use_xray_dataset:
#         if skip_processing:
#             xray_df = ld.load_csv(processed_labels_path)
#             print(xray_df.shape)
#             print('Cardiomegaly label division')
#             print(xray_df['Cardiomegaly'].value_counts())
#         else:
#             label_df = ld.get_classification_labels(classication_labels_path, False)
#             processed_df = ld.preprocess_labels(label_df, image_path)
#             xray_df = ld.couple_location_labels(localization_labels_path, processed_df, ld.PATCH_SIZE, results_path)
#     # print(xray_df.shape)
#     # print("Splitting data ...")
# #
# # #
# # def load_data_cv_mura():
# #     df_train, df_val = split_data_cv(df, splits_nr, current_split, random_seed, diagnose_col, ratio_to_keep=None)
