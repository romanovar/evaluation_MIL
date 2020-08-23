import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from stability.preprocessing import binarize_predictions
from stability.stability_scores import compute_additional_scores_kappa
import numpy as np


def compute_auc_1class(labels_all_classes, img_predictions_all_classes):
    auc_score = roc_auc_score(labels_all_classes[:, ], img_predictions_all_classes[:, ])
    return auc_score


def get_image_index_xray(image_link_collection, image_ind):
    return image_link_collection[image_ind][-16:-4]


def get_image_index_mura(image_link_collection, image_index):
    patient_id_image_id = (image_link_collection[image_index].partition("/patient"))[-1]
    # '1_positive/image1.png' to image1.png
    image_id = patient_id_image_id.partition("/study")[-1].partition("/")[-1]
    patient_id = patient_id_image_id.partition("/study")[0]
    return patient_id+'_'+image_id


def get_image_index(xray, image_link_collection, image_index):
    if xray:
        return get_image_index_xray(image_link_collection, image_index)
    else:
        return get_image_index_mura(image_link_collection, image_index)


def init_csv():
    df = pd.DataFrame()
    df['Image_ind'] = 0
    df['po'] = -100
    df['K'] = -100
    df['pos_K'] = -100
    df['p_pos'] = -100
    df['p_neg'] = -100
    df['pos-neg'] = -100
    df['f1-f2'] = -100
    return df


def save_kappa_scores_csv(img_ind, po, aiou, apj,  p_pos, p_neg, diff_p_pos_p_neg, diff_f1_f2, diff_g1_g2,
                          unique_file_identifier, res_path):
    df = init_csv()
    df['Image_ind'] = img_ind
    df['po'] = po
    df['K'] = aiou
    df['pos_K'] = apj
    df['p_pos'] = p_pos
    df['p_neg'] = p_neg
    df['pos-neg'] = diff_p_pos_p_neg
    df['f1-f2'] = diff_f1_f2
    df['g1-g2'] = diff_g1_g2
    df.to_csv(res_path+'additional_scores_kappa'+unique_file_identifier+'.csv')


def save_mean_stability(img_ind, jacc, corr_jacc, iou, spearman, res_path, file_identifier, dice=None):
    df = pd.DataFrame()
    column_names = ['Image_ind', 'Mean positive Jaccard', 'Mean corrected positive Jaccard',
                    'Mean corrected IoU', 'Mean Spearman']

    values = [img_ind,  jacc, corr_jacc, iou, spearman]
    if dice is not None:
        column_names.extend(['Mean dice'])
        values.extend([dice])

    for column, column_values in zip(column_names, values):
        df[column] = column_values

    df2 = calculate_aggregated_performance(column_names, 'mean', values)
    df3 = calculate_aggregated_performance(column_names, 'stand dev', values)

    df = df.append(df2)
    df = df.append(df3)
    df.to_csv(res_path+'mean_stability_'+file_identifier+'.csv')


def save_additional_kappa_scores_forthreshold(thres, raw_pred_coll, img_ind, corr_iou_coll,
                                              corr_pos_jacc_coll, res_path):
    binary_predictions_coll = []
    for raw_pred in raw_pred_coll:
        binary_predictions = binarize_predictions(raw_pred, threshold=thres)
        binary_predictions_coll.append(binary_predictions)

    for bin_pred_ind in range(0, len(binary_predictions_coll)):
        for bin_pred_ind2 in range(0, len(binary_predictions_coll)):
            corr_iou = corr_iou_coll[bin_pred_ind, bin_pred_ind2, :]
            corr_pos_jacc = corr_pos_jacc_coll[bin_pred_ind, bin_pred_ind2, :]
            po, p_pos, p_neg, diff_p_pos_p_neg, diff_f1_f2,  diff_g1_g2 =\
                compute_additional_scores_kappa(binary_predictions_coll[bin_pred_ind],
                                                binary_predictions_coll[bin_pred_ind2])
            save_kappa_scores_csv(img_ind[0], po, corr_iou, corr_pos_jacc, p_pos, p_neg, diff_p_pos_p_neg,
                                  diff_f1_f2, diff_g1_g2, str(bin_pred_ind) + '_'+str(bin_pred_ind2), res_path)


def calculate_aggregated_performance(columns, operation, stability_values):
    operation_dict = {'mean': np.mean,
                      'stand dev':np.std}
    aggregated_values = operation_dict[operation](stability_values[1:], axis=1)
    values = [operation]
    values.extend(aggregated_values)

    df2 = pd.DataFrame([values], columns=columns)
    return df2


def get_matrix_total_nans_stability_score(stab_index_collection, total_images_collection, normalize):
    nan_matrix = np.count_nonzero(np.isnan(np.array(stab_index_collection).
                                           reshape(5, 5, len(total_images_collection[0]))), axis=-1)
    if normalize:
        return nan_matrix / len(total_images_collection[0])
    else:
        return nan_matrix


def get_nonduplicate_scores(total_images, models_nr, stability_score_coll):
    pairwise_stability_all_images = []
    # total_images = inst_labels[0].shape[0]
    # FOR EACH IMAGE, THE PREDICTIONS OF EACH CLASSIFIERS ARE COMPARED WITH THE WHOLE BAG AND AUC IS COMPUTED
    for image_ind in range(0, total_images):
        image_stab_scores = []
        for classifier_ind in range(0, models_nr):
            stab_scores = stability_score_coll[classifier_ind, classifier_ind + 1:, image_ind]
            image_stab_scores = np.concatenate((image_stab_scores, stab_scores))
        pairwise_stability_all_images.append(image_stab_scores)
    # TOTAL_IMAGES x 10 combinations of stability
    stability_res = np.array(pairwise_stability_all_images)

    return stability_res


# todo: delete if not used
def compute_ap(inst_labels, inst_pred):
    image_ap_collection_all_classifiers = []
    # pairwise_stability_all_images = []
    total_images = inst_labels[0].shape[0]
    # instance auc
    # FOR EACH IMAGE, THE PREDICTIONS OF EACH CLASSIFIERS ARE COMPARED WITH THE WHOLE BAG AND AUC IS COMPUTED
    for image_ind in range(0, total_images):
        all_instances_labels = inst_labels[0].reshape(total_images, -1)
        ap_collection = []

        for classifier_ind in range(0, 5):
            inst_predictions_classifier = inst_pred[classifier_ind].reshape(total_images, -1)
            ap_classifiers = average_precision_score(all_instances_labels[image_ind],
                                                     inst_predictions_classifier[image_ind])
            ap_collection.append(ap_classifiers)

        image_ap_collection_all_classifiers.append(ap_collection)

    # TOTAL_IMAGES x TOTAL_CLASSIFIERS
    ap_res = np.array(image_ap_collection_all_classifiers)
    return ap_res