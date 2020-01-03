import numpy as np
import pandas as pd


def load_prediction_files(inst_lab_prefix, ind_prefix, inst_pred_prefix, bag_lab_prefix, bag_pred_prefix,
                          dataset_name, predictions_path):
    labels = np.load(predictions_path + inst_lab_prefix + dataset_name, allow_pickle=True)
    image_indices = np.load(predictions_path+ind_prefix+dataset_name, allow_pickle=True)
    predictions = np.load(predictions_path+inst_pred_prefix+dataset_name, allow_pickle=True)

    bag_labels = np.load(predictions_path+bag_lab_prefix+(dataset_name[:-4])+'_as_production0'+ '.npy' ,
                         allow_pickle=True)
    bag_predictions = np.load(predictions_path+bag_pred_prefix+dataset_name[:-4]+'_as_production0'+ '.npy' ,
                              allow_pickle=True)

    return labels, image_indices, predictions, bag_labels, bag_predictions


def filter_bbox_image_ind(labels):
    bbox_ind_col2 = []
    sum_all = np.sum(np.reshape(labels, (labels.shape[0], 16 * 16 * 1)), axis=1)
    print("**************")
    for el_ind in range(0, sum_all.shape[0]):
        if 0 < sum_all[el_ind] < 256:
            print(el_ind)
            bbox_ind_col2.append(el_ind)
    return bbox_ind_col2


def binarize_predictions(raw_prediction, threshold):
    return np.array(raw_prediction > threshold, dtype=int)


def calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2, P=16):
    sum_preds = bin_pred1 + bin_pred2
    n11_mask = np.array(sum_preds > 1, dtype=int)
    n00_mask = np.array(sum_preds == 0, dtype=int)

    # REMOVES ELEMENTS EQUAL 0, SO ONLY 1 are left
    pred1_n1_mask2 = np.ma.masked_equal(bin_pred1, 0)
    pred1_n0_mask2 = np.ma.masked_equal(bin_pred1, 1)

    pred2_n1_mask2 = np.ma.masked_equal(bin_pred2, 0)
    pred2_n0_mask2 = np.ma.masked_equal(bin_pred2, 1)

    n10_2 = np.sum((pred1_n1_mask2 + pred2_n0_mask2).reshape(-1, P*P), axis=1)
    n01_2 = np.sum((pred1_n0_mask2 + pred2_n1_mask2).reshape(-1, P*P), axis=1)
    n10 = np.asarray(n10_2)
    n01 = np.asarray(n01_2)

    n11 = np.sum(n11_mask.reshape((n11_mask.shape[0], P*P)), axis=1)
    n00 = np.sum(n00_mask.reshape((n00_mask.shape[0], P*P)), axis=1)
    return n00, n10, n01, n11



def load_predictions(set_name1, set_name2, predict_res_path):
    patch_labels_prefix = 'patch_labels_'
    img_ind_prefix = 'image_indices_'
    raw_pred_prefix = 'predictions_'
    bag_lab_prefix = 'image_labels_'
    bag_pred_prefix = 'image_predictions_'

    df_stability = pd.DataFrame()
    df_auc = pd.DataFrame()

    all_labels_1, all_image_ind_1, all_raw_predictions_1,  \
    all_bag_labels, all_bag_predictions = load_prediction_files(patch_labels_prefix, img_ind_prefix,
                                                                                 raw_pred_prefix,
                                                                                 bag_lab_prefix, bag_pred_prefix,
                                                                                 set_name1, predict_res_path)
    all_labels_95, all_image_ind_95, all_raw_predictions_95, \
    all_bag_labels_95, all_bag_predictions_95 = load_prediction_files(patch_labels_prefix, img_ind_prefix,
                                                                                    raw_pred_prefix,
                                                                                    bag_lab_prefix, bag_pred_prefix,
                                                                                    set_name2, predict_res_path)
    return  all_labels_1, all_image_ind_1, all_raw_predictions_1, all_bag_labels, all_bag_predictions,\
            all_labels_95, all_image_ind_95, all_raw_predictions_95, all_bag_labels_95, all_bag_predictions_95



def load_predictions_v2(classifier_name_list, predict_res_path):
    patch_labels_prefix = 'patch_labels_'
    img_ind_prefix = 'image_indices_'
    raw_pred_prefix = 'predictions_'
    bag_predictions = 'image_predictions_'
    bag_labels = 'image_labels_'
    all_labels = []
    all_image_ind = []
    all_raw_predictions = []
    all_bag_predictions = []
    all_bag_labels = []
    for classifier in classifier_name_list:
        all_labels_classifier, all_image_ind_classifier, \
        all_raw_predictions_classifier, all_bag_labels_class, all_bag_predictions_class = \
            load_prediction_files(patch_labels_prefix, img_ind_prefix, raw_pred_prefix, bag_labels,
                                  bag_predictions, classifier, predict_res_path)
        # all_labels_classifier, all_image_ind_classifier, \
        # all_raw_predictions_classifier = load_prediction_files(patch_labels_prefix, img_ind_prefix,
        #                                                        raw_pred_prefix, bag_labels,
        #                                                        bag_predictions,
        #                                                        classifier, predict_res_path)

        all_labels.append(all_labels_classifier)
        all_image_ind.append(all_image_ind_classifier)
        all_raw_predictions.append(all_raw_predictions_classifier)
        all_bag_predictions.append(all_bag_predictions_class)
        all_bag_labels.append(all_bag_labels_class)
    return all_labels, all_image_ind, all_raw_predictions, all_bag_labels, all_bag_predictions


def indices_segmentation_images(all_labels_1, all_labels_2):
    bbox_indices1 = filter_bbox_image_ind(all_labels_1)
    bbox_indices2 = filter_bbox_image_ind(all_labels_2)
    assert bbox_indices1 == bbox_indices2, "Error, bbox images should be equal " \
                                            "in both cases"
    print("Total images found with segmenation is: " + str(len(bbox_indices2)))
    return bbox_indices1, bbox_indices2


def indices_segmentation_images_v2(all_labels_collection):
    bbox_ind_collection =[]
    for all_labels in all_labels_collection:
        bbox_indices = filter_bbox_image_ind(all_labels)
        bbox_ind_collection.append(bbox_indices)

    for bbox_ind in bbox_ind_collection:
        for bbox_ind2 in bbox_ind_collection:
            assert bbox_ind == bbox_ind2, "Error, bbox images should be equal " \
                                                    "in both cases"
    print("Total images found with segmenation is: " + str(len(bbox_ind_collection[0])))
    return bbox_ind_collection


def indices_positive_images(bag_labels_collection):
    positive_img_ind_collection =[]
    for img_labels in bag_labels_collection:
        # get indices of elements equal to 1
        pos_img_indices = np.where(img_labels == 1)[0]
        positive_img_ind_collection.append(pos_img_indices)

    for pos_ind in positive_img_ind_collection:
        for pos_ind2 in positive_img_ind_collection:
            assert (pos_ind == pos_ind2).all(), "Error, positive images should be equal " \
                                                    "in both cases"
    print("Total positive images found is: " + str(len(positive_img_ind_collection[0])))
    return positive_img_ind_collection


def filter_predictions_files_segmentation_images(all_labels_1, all_image_ind_1, all_raw_predictions_1, bbox_indices1,
                                                 all_labels_2, all_image_ind_2, all_raw_predictions_2, bbox_indices2):
    labels_1, image_ind_1, raw_predictions_1 = all_labels_1[bbox_indices1], all_image_ind_1[bbox_indices1], \
                                               all_raw_predictions_1[bbox_indices1]
    labels_2, image_ind_2, raw_predictions_2 = all_labels_2[bbox_indices2], all_image_ind_2[bbox_indices2], \
                                               all_raw_predictions_2[bbox_indices2]
    return  labels_1, image_ind_1, raw_predictions_1, labels_2, image_ind_2, raw_predictions_2


def filter_predictions_files_on_indeces(all_labels_coll, all_image_ind_coll, all_raw_predictions_coll,
                                        all_bag_predictions_coll, all_bag_labels_coll, bbox_ind_coll):
    '''

    :param all_labels_coll:
    :param all_image_ind_coll:
    :param all_raw_predictions_coll:
    :param bbox_ind_coll:
    :return: Returns only the images, labels and raw predictions of images with bounding boxes
    '''
    bbox_img_labels_coll = []
    bbox_img_ind_coll = []
    bbox_img_raw_predictions = []
    bbox_img_bag_predictions = []
    bbox_img_bag_labels = []
    assert len(bbox_ind_coll) == len(all_labels_coll) == len(all_image_ind_coll) == len(all_raw_predictions_coll), \
        "The lists do not have the same length"

    for el_ind in range(0, len(all_labels_coll)):
        bbox_ind = bbox_ind_coll[el_ind]
        labels, image_ind, raw_predictions, bag_predictions, bag_labels = (all_labels_coll[el_ind])[bbox_ind], \
                                                                          (all_image_ind_coll[el_ind])[bbox_ind], \
                                                                          (all_raw_predictions_coll[el_ind])[bbox_ind],\
                                                                          (all_bag_predictions_coll[el_ind])[bbox_ind], \
                                                                          (all_bag_labels_coll[el_ind])[bbox_ind]

        bbox_img_labels_coll.append(labels)
        bbox_img_ind_coll.append(image_ind)
        bbox_img_raw_predictions.append(raw_predictions)
        bbox_img_bag_predictions.append(bag_predictions)
        bbox_img_bag_labels.append(bag_labels)
        print("""""""""""""""""""")
        print(el_ind)
        print(image_ind)
        assert (bbox_img_ind_coll[0] == image_ind).all(), "bbox image index are different or in different order"
    return bbox_img_labels_coll, bbox_img_ind_coll, bbox_img_raw_predictions,\
           bbox_img_bag_labels, bbox_img_bag_predictions

#
# def filter_predictions_files_positive_images(all_labels_coll, all_image_ind_coll, all_raw_predictions_coll,
#                                             all_bag_predictions_coll, all_bag_labels_coll, positive_img_ind_coll):
#     '''
#
#     :param all_labels_coll:
#     :param all_image_ind_coll:
#     :param all_raw_predictions_coll:
#     :param bbox_ind_coll:
#     :return: Returns only the images, labels and raw predictions of images with bounding boxes
#     '''
#     bbox_img_labels_coll = []
#     bbox_img_ind_coll = []
#     bbox_img_raw_predictions = []
#     bbox_img_bag_predictions = []
#     bbox_img_bag_labels = []
#     assert len(positive_img_ind_coll) == len(all_labels_coll) == \
#            len(all_image_ind_coll) == len(all_raw_predictions_coll), \
#         "The lists do not have the same length"
#
#     for el_ind in range(0, len(all_labels_coll)):
#         pos_ind = positive_img_ind_coll[el_ind]
#         labels, image_ind, raw_predictions, bag_predictions, bag_labels = (all_labels_coll[el_ind])[pos_ind], \
#                                                                           (all_image_ind_coll[el_ind])[pos_ind], \
#                                                                           (all_raw_predictions_coll[el_ind])[pos_ind],\
#                                                                           (all_bag_predictions_coll[el_ind])[pos_ind], \
#                                                                           (all_bag_labels_coll[el_ind])[pos_ind]
#
#         bbox_img_labels_coll.append(labels)
#         bbox_img_ind_coll.append(image_ind)
#         bbox_img_raw_predictions.append(raw_predictions)
#         bbox_img_bag_predictions.append(bag_predictions)
#         bbox_img_bag_labels.append(bag_labels)
#         print("""""""""""""""""""")
#         print(el_ind)
#         print(image_ind)
#         assert bbox_img_ind_coll[0].all() == image_ind.all(), "bbox image index are different or in different order"
#     return bbox_img_labels_coll, bbox_img_ind_coll, bbox_img_raw_predictions,\
#            bbox_img_bag_labels, bbox_img_bag_predictions
