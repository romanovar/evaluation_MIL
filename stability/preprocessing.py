import numpy as np


def load_model_prediction_from_file(inst_lab_prefix, ind_prefix, inst_pred_prefix, bag_lab_prefix, bag_pred_prefix,
                                    bbox_prefix, dataset_name, predictions_path):
    """
    Loads all prediction files associated WITH SPECIFIC MODEL
    :param inst_lab_prefix: file prefix for instance labels file
    :param ind_prefix: file prefix for image index
    :param inst_pred_prefix: file prefix for instance predictions
    :param bag_lab_prefix: file prefix for bag labels
    :param bag_pred_prefix: file prefix for bag predictions
    :param dataset_name: dataset identifier
    :param predictions_path: directory of the prediction files
    :return: Instance labels, unique sample names, instance predictions, bag labels and bag predictions for each sample
    """
    labels = np.load(predictions_path + inst_lab_prefix + dataset_name + '.npy', allow_pickle=True)
    image_indices = np.load(predictions_path+ind_prefix + dataset_name + '.npy', allow_pickle=True)
    predictions = np.load(predictions_path+inst_pred_prefix+ dataset_name + '.npy', allow_pickle=True)

    bag_labels = np.load(predictions_path+bag_lab_prefix + dataset_name + '.npy',
                         allow_pickle=True)
    bag_predictions = np.load(predictions_path+bag_pred_prefix + dataset_name + '.npy',
                              allow_pickle=True)
    bbox_available = np.load(predictions_path+bbox_prefix + dataset_name + '.npy', allow_pickle=True)
    return labels, image_indices, predictions, bag_labels, bag_predictions, bbox_available


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


def load_filter_dice_scores(classifiers_list, segm_img_index, predict_res_path):
    dice_scores_coll = []
    for classifier in classifiers_list:
        dice = np.load(predict_res_path + 'dice_' + classifier+'.npy', allow_pickle=True)
        dice_scores_coll.append(dice[segm_img_index[0]])
    return dice_scores_coll


def load_predictions(classifier_name_list, predict_res_path):
    '''
    Loads the predictions from a list of models of the same architecture.
    The predictions are used to compare stability between these models.
    :param classifier_name_list: list of several models trained on slightly different subsets
    :param predict_res_path:
    :return: all_labels, all_image_ind, all_raw_predictions, all_bag_labels, all_bag_predictions as lists with each
    element in the list correspond to a list of classifier predictions
    '''

    all_labels = []
    all_image_ind = []
    all_raw_predictions = []
    all_bag_predictions = []
    all_bag_labels = []
    all_bbox = []
    for classifier in classifier_name_list:
        all_labels_classifier, all_image_ind_classifier, \
        all_raw_predictions_classifier, all_bag_labels_class, all_bag_predictions_class, all_bbox_classifier = \
            load_model_prediction_from_file(dataset_name=classifier, predictions_path=predict_res_path,
                                            inst_lab_prefix= 'patch_labels_',
                                            ind_prefix='image_indices_', inst_pred_prefix= 'predictions_',
                                            bag_lab_prefix= 'image_labels_', bag_pred_prefix='image_predictions_',
                                            bbox_prefix='bbox_present_')

        all_labels.append(all_labels_classifier)
        all_image_ind.append(all_image_ind_classifier)
        all_raw_predictions.append(all_raw_predictions_classifier)
        all_bag_predictions.append(all_bag_predictions_class)
        all_bag_labels.append(all_bag_labels_class)
        all_bbox.append(all_bbox_classifier)
    return all_labels, all_image_ind, all_raw_predictions, all_bag_labels, all_bag_predictions, all_bbox


def indices_segmentation_images(all_labels_collection):
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


def filter_predictions_files_on_indices(all_labels_coll, all_image_ind_coll, all_raw_predictions_coll,
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


def load_and_filter_predictions(config, classifiers, only_segmentation_images, only_positive_images):
    '''
    Loads prediction files and filters on specific samples, which are used to compute stability
    :param config: config file
    :param classifiers: list with all classifier names
    :param only_segmentation_images: True: analysis is done only on images with segmentation,
                                    False: analysis is done on all images
    :param only_positive_images: this parameter is considered only if only_segmentation_images is FALSE
                True: analysis done on image with positive label
                False: analysis is done on all images
    :return: Returns the suitable rows of the subset desired
    '''

    prediction_results_path = config['prediction_results_path']

    instance_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
    bag_predictions_collection, _ = load_predictions(classifiers, prediction_results_path)

    if only_segmentation_images:
        filtered_idx_collection = indices_segmentation_images(instance_labels_collection)
        identifier = "_segmented_img"
        instance_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
        bag_predictions_collection = filter_predictions_files_on_indices(instance_labels_collection, image_index_collection,
                                                                         raw_predictions_collection,
                                                                         bag_predictions_collection, bag_labels_collection,
                                                                         filtered_idx_collection)
        return instance_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
               bag_predictions_collection, identifier

    elif only_positive_images:
        filtered_idx_collection = indices_positive_images(bag_labels_collection)
        identifier = "_pos_img"
        instance_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
        bag_predictions_collection = filter_predictions_files_on_indices(instance_labels_collection, image_index_collection,
                                                                         raw_predictions_collection,
                                                                         bag_predictions_collection, bag_labels_collection,
                                                                         filtered_idx_collection)
        return instance_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
               bag_predictions_collection, identifier
    else:
        identifier = "_all_img"
        return instance_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
               bag_predictions_collection, identifier


def filter_segmentation_images_bbox_file(config, classifiers):
    prediction_results_path = config['prediction_results_path']

    image_labels_collection, image_index_collection, raw_predictions_collection, bag_labels_collection, \
    bag_predictions_collection, bbox_collection = load_predictions(classifiers, prediction_results_path)

    for model_idx in range(1, len(bbox_collection)):
        assert (bbox_collection[model_idx] == bbox_collection[model_idx-1]).all(), "bbox files are not equal!"
    idx_to_filter_coll = []
    for model_idx in range(len(bbox_collection)):
        idx_to_filter_coll.append(np.where(bbox_collection[model_idx]==True)[0])

    return idx_to_filter_coll
