import pandas as pd
from sklearn.metrics import roc_auc_score

from stability.preprocessor.preprocessing import binarize_predictions
from stability.stability_2classifiers.stability_scores import compute_additional_scores_kappa
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


def save_mean_stability(img_ind, jacc, corr_jacc, iou, spearman, res_path, file_identifier):
    df = pd.DataFrame()
    df['Image_ind'] = 0
    df['Mean Jaccard'] = -100
    df['Mean corrected jaccard'] = -100
    df['mean IoU'] = -100
    df['Mean Spearman'] = -100

    df['Image_ind'] = img_ind
    df['Mean Jaccard'] = jacc
    df['Mean corrected jaccard'] = corr_jacc
    df['mean IoU'] = iou
    df['Mean Spearman'] = spearman

    df.to_csv(res_path+'mean_stability_'+file_identifier+'.csv')


def save_mean_stability_auc(img_ind, auc, corr_jacc, spearman, res_path, file_identifier, ap):
    df = pd.DataFrame()
    df['Image_ind'] = 0
    df['Mean Instance AUC'] = -100
    df['Standard deviation AUC'] = -100
    df['Mean corrected jaccard'] = -100
    df['Mean Spearman'] = -100

    df['Image_ind'] = img_ind
    df['Mean Instance AUC'] = np.mean(auc, axis=1)
    df['Standard deviation AUC'] = np.std(auc, axis=1)
    df['Mean corrected jaccard'] = corr_jacc
    df['Mean Spearman'] = spearman
    df['Mean AP'] = np.mean(ap, axis=1)
    df.to_csv(res_path + 'mean_stability_inst_auc_' + file_identifier + '.csv')


def save_mean_dice(img_ind,dice, accuracy, res_path, file_identifier):
    '''

    :param img_ind: list with image names/ indeces
    :param dice: list with dice scores for each image
    :param accuracy: list with accuracy for each image
    :param res_path: path to save generated file
    :param file_identifier: unique name to save the file
    :return: save s a .csv file with the dice score and accuracy from all classifiers for each image
    '''
    df = pd.DataFrame()
    df['Image_ind'] = 0
    df['Mean DICE'] = -100
    df['STD DICE'] = -100
    df['Mean Accuracy'] = -100

    df['Image_ind'] = img_ind
    df['Mean DICE'] = np.mean(dice, axis=1)
    df['STD DICE'] = np.std(dice, axis=1)
    df['Mean Accuracy'] = np.mean(accuracy, axis=1)
    df.to_csv(res_path + 'mean_dice_inst_' + file_identifier + '.csv')


