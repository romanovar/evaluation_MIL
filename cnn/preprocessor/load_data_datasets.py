import cnn.preprocessor.load_data as ld
import cnn.preprocessor.load_data_mura as ldm


def load_xray14(config):
    skip_processing = config['skip_processing_labels']
    image_path = config['image_path']
    processed_labels_path = config['processed_labels_path']
    classication_labels_path = config['classication_labels_path']
    localization_labels_path = config['localization_labels_path']
    generated_images_path = config['generated_images_path']
    results_path = config['results_path']


    if skip_processing:
        xray_df = ld.load_csv(processed_labels_path)
        print('Cardiomegaly label division')
        print(xray_df['Cardiomegaly'].value_counts())
    else:
        label_df = ld.get_classification_labels(classication_labels_path, False)
        processed_df = ld.preprocess_labels(label_df, image_path)
        xray_df = ld.couple_location_labels(localization_labels_path, processed_df, ld.PATCH_SIZE, results_path)
    print(xray_df.shape)
    print("Splitting data ...")
    init_train_idx, df_train_init, df_val, \
    df_bbox_test, df_class_test, df_bbox_train = ld.get_train_test(xray_df, random_state=1, do_stats=False,
                                                                   res_path = generated_images_path,
                                                                   label_col = 'Cardiomegaly')
    df_train=df_train_init
    print('Training set: '+ str(df_train_init.shape))
    print('Validation set: '+ str(df_val.shape))
    print('Localization testing set: '+ str(df_bbox_test.shape))
    print('Classification testing set: '+ str(df_class_test.shape))

    # df_train = keras_utils.create_overlap_set_bootstrap(df_train_init, 0.9, seed=2)
    init_train_idx = df_train['Dir Path'].index.values

    # new_seed, new_tr_ind = ld.create_overlapping_test_set(init_train_idx, 1, 0.95,0.85, xray_df)
    # print(new_tr_ind)
    return df_train, df_val, df_bbox_test, df_class_test


def load_mura(config):
    mura_train_img_path = config['mura_train_img_path']
    mura_train_labels_path = config['mura_train_labels_path']
    mura_test_img_path = config['mura_test_img_path']
    mura_test_labels_path = config['mura_test_labels_path']
    processed_train_labels_path = config['mura_processed_train_labels_path']
    processed_test_labels_path = config['mura_processed_test_labels_path']

    skip_processing = config['skip_processing_labels']

    class_name = config['mura_class']
    ldm.check_validity_class(class_name)
    if skip_processing:
        df_train_val = ld.load_csv(processed_train_labels_path)
        test_df_all_classes = ld.load_csv(processed_test_labels_path)

    else:
        end_class = mura_train_img_path.find('MURA-v1.1')
        mura_folder_root = mura_train_img_path[0:end_class]
        print(mura_folder_root)
        df_train_val = ldm.get_save_processed_df(mura_train_labels_path, mura_train_img_path, mura_folder_root, "train_mura")
        test_df_all_classes = ldm.get_save_processed_df(mura_test_labels_path, mura_test_img_path, mura_folder_root,
                                                        "test_mura")

    _, _, train_df_all_classes, val_df_all_classes = ldm.split_train_val_set(df_train_val)
    df_train, df_val, df_test = ldm.filter_all_set_for_class(train_df_all_classes, val_df_all_classes,
                                                             test_df_all_classes, class_name)
    df_train_final = ld.keep_index_and_1diagnose_columns(df_train, 'instance labels')
    df_val_final = ld.keep_index_and_1diagnose_columns(df_val, 'instance labels')
    df_test_final = ld.keep_index_and_1diagnose_columns(df_test, 'instance labels')


    print('Training set: ' + str(df_train_final.shape))
    print('Validation set: ' + str(df_val_final.shape))
    # print('Localization testing set: '+ str(df_bbox_test.shape))
    print('Classification testing set: ' + str(df_test_final.shape))
    return df_train_final, df_val_final, df_test_final