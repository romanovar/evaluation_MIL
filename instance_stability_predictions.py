from pathlib import Path

import cv2
import matplotlib
import yaml
import argparse
import os
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from behavior_stability_score_visualizations import plot_line_graph
from keras_utils import get_image_index_from_pathstring

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.load(ymlfile)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str,
                    help='Provide the file path to the configuration')

args = parser.parse_args()
config = load_config(args.config_path)

skip_processing = config['skip_processing_labels']
image_path = config['image_path']
classication_labels_path = config['classication_labels_path']
localization_labels_path = config['localization_labels_path']
results_path = config['results_path']
generated_images_path = config['generated_images_path']
processed_labels_path = config['processed_labels_path']
prediction_results_path = config['prediction_results_path']
train_mode = config['train_mode']
test_single_image = config['test_single_image']
trained_models_path = config['trained_models_path']
stability_res_path = config['stability_results']


def load_prediction_files(lab_prefix, ind_prefix, pred_prefix, dataset_name, predictions_path):
    labels = np.load(predictions_path+lab_prefix+dataset_name, allow_pickle=True)
    image_indices = np.load(predictions_path+ind_prefix+dataset_name, allow_pickle=True)
    predictions = np.load(predictions_path+pred_prefix+dataset_name, allow_pickle=True)
    return labels, image_indices, predictions


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


def positive_Jaccard_index_batch(bin_pred1, bin_pred2, P):
    """
    :param bin_pred1: raw predictions of all bbox images
    :param bin_pred2: raw predictions of another subset of all bbox images
    :return: array with positive  jaccard index for
    """
    sum_preds = bin_pred1 + bin_pred2
    n11_mask = np.array(sum_preds > 1, dtype=int)
    n10_n01_mask = np.array(sum_preds ==1, dtype=int)

    n11 = np.sum(n11_mask.reshape((n11_mask.shape[0], P*P)), axis=1)
    n10_n01 = np.sum(n10_n01_mask.reshape((n10_n01_mask.shape[0], P*P)), axis=1)
    np.seterr(divide='ignore', invalid='ignore')
    pos_jaccard_dist = n11/(n11+n10_n01)
    return pos_jaccard_dist


def calculate_spearman_rank_coefficient(pred1, pred2):
    spearman_corr_coll = []
    assert pred1.shape[0] == pred2.shape[0], "Ensure the predictions have same shape!"
    for obs in range(0, pred1.shape[0]):
        rank_image1 = rankdata(pred1.reshape(-1, 16*16*1)[obs])
        rank_image2 = rankdata(pred2.reshape(-1, 16*16*1)[obs])
        rho, pval = spearmanr(rank_image1, rank_image2)
        spearman_corr_coll.append(rho)
    return spearman_corr_coll


def calculate_spearman_rank_coefficient_v2(scores):
    '''
    Calculating spearmancorrelation coefficient between the stability scores,
    not betwween predicitions
    :param pred1: stability_score1
    :param pred2:  stability_score2
    :return:
    '''

    assert len(scores) >= 2, "List of indices is too short"
    for ind in range(1, len(scores)):
        if ind ==1:
            ranks = np.append([rankdata(scores[ind-1])], [rankdata(scores[ind])], axis=0)
        else:
            ranks = np.append(ranks, [rankdata((scores[ind]))], axis=0)
    rho, pval =spearmanr(ranks, axis=1)

    return rho


def draw_heatmap(df, labels, ax, font_size_annotations, drop_duplicates):
    cmap = sns.cubehelix_palette(8, as_cmap=True)
    if drop_duplicates:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        htmp = sns.heatmap(df, cmap=cmap, mask=mask, square=True, annot=True,
                           annot_kws={"size": font_size_annotations}, xticklabels=labels, yticklabels=labels,
                           linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        htmp = sns.heatmap(df, cmap=cmap, square=True, annot=True,
                           annot_kws={"size": font_size_annotations}, xticklabels=labels, yticklabels=labels,
                           linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    return htmp


# https://stackoverflow.com/questions/34739950/how-to-save-a-plot-in-seaborn-with-python
def visualize_correlation_heatmap(df, res_path, img_ind, labels, dropDuplicates = True):
    # Set background color / chart style
    sns.set_style(style='white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue
    cmap = sns.cubehelix_palette(8, as_cmap=True)
    # cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.set(font_scale=2)
    # Draw correlation plot with or without duplicates
    # if dropDuplicates:
        # htmp = sns.heatmap(df, mask=mask, cmap=cmap,
        #             square=True, annot=True, annot_kws={"size":18}, xticklabels=labels, yticklabels=labels,
        #             linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    htmp = draw_heatmap(df, labels, ax, 18, dropDuplicates)
    # else:
        # htmp = sns.heatmap(df, cmap=cmap,
        #             square=True, annot=True, annot_kws={"size":18}, xticklabels=labels, yticklabels=labels,
        #             linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
        # htmp = draw_heatmap(df, labels, ax, 18)
    plt.show()
    htmp.figure.savefig(res_path + 'correlation_' + img_ind + '.jpg', bbox_inches='tight')
    plt.close()
    return htmp


def combine_correlation_heatmaps_next_to_each_other(df1, df2, subtitle1, subtitle2, labels, res_path, img_ind, drop_duplicates):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    sns.set_style(style='white')
    sns.set(font_scale=1)

    ax1 = plt.subplot(1, 2, 1)
    htmp = draw_heatmap(df1, labels, ax1, 10, drop_duplicates)
    ax1.set_title('Stability Index: ' + subtitle1, {'fontsize': 9})

    ax2= plt.subplot(1, 2, 2)
    htmp = draw_heatmap(df2, labels, ax2, 10, drop_duplicates)
    ax2.set_title('Stability Index: ' + subtitle2, {'fontsize': 9})
    plt.show()
    htmp.figure.savefig(res_path + 'correlation_combo_' + img_ind + '.jpg', bbox_inches='tight')
    plt.close()
    return htmp


def calculate_pearson_coefficient_batch(raw_pred1, raw_pred2):
    correlation_coll = []
    assert raw_pred1.shape == raw_pred2.shape, "Predictions don't have same shapes"

    for ind in range(0, raw_pred1.shape[0]):
        corr_coef = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
                    raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind])
        correlation_coll.append(corr_coef[0,1])
        corr_coef2 = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
                    raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind], rowvar=False)
        assert corr_coef[0, 1]==corr_coef2[0, 1], "think on the dimensions of the correlation computed "

    return correlation_coll


def calculate_AUC_batch(pred, labels):
    auc_coll = []
    assert pred.shape == labels.shape, "Labels and predictions not from the same shape"
    for ind in range(0, pred.shape[0]):
        auc_single_image = roc_auc_score(labels[ind, :, :, :].reshape(16*16*1), pred[ind, :, :, :].reshape(16*16*1))
        auc_coll.append(auc_single_image)
    return auc_coll


def calculate_IoU(bin_pred1, bin_pred2):
    sum_preds = bin_pred1 + bin_pred2
    n11_mask = np.array(sum_preds > 1, dtype=int)
    n00_mask = np.array(sum_preds == 0, dtype=int)
    n10_n01_mask = np.array(sum_preds == 1, dtype=int)

    n11 = np.sum(n11_mask.reshape((n11_mask.shape[0], 16 * 16 * 1)), axis=1)
    n00 = np.sum(n00_mask.reshape((n00_mask.shape[0], 16*16*1)), axis=1)
    n10_n01 = np.sum(n10_n01_mask.reshape((n10_n01_mask.shape[0], 16 * 16 * 1)), axis=1)
    np.seterr(divide='ignore', invalid='ignore')

    iou_score = (n11+n00) / (n11 + n10_n01+n00)
    return iou_score


def overlap_coefficient(bin_pred1, bin_pred2, P):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2, P)
    min_n01_n10 = np.minimum(n10, n01)
    return n11/(min_n01_n10 + n11)


def corrected_overlap_coefficient(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)

    min_n01_n10 = np.minimum(n10, n01)
    # assert (n11+n01) == np.sum(np.ma.masked_equal(bin_pred2, 0).reshape(-1, 16*16*1), axis=1),\
    #     "Error with computing the positive instances "
    # assert (n11+n10) == np.sum(np.ma.masked_equal(bin_pred1, 0).reshape(-1, 16*16*1), axis=1),\
    #     "Error with computing the positive instances "
    N = n00 + n11 + n10 + n01
    expected_overlap = (n11+n01)*(n11+n10)/N
    # 0/0 -> nan so convert nan to 0
    # corrected_score = np.nan_to_num((n11 - expected_overlap)/(np.minimum((n11+n01), (n11+n10)) - expected_overlap))
    # corrected_score2 = np.nan_to_num((n00*n11 - n10*n01)/((min_n01_n10+n11)*(min_n01_n10 + n00)))
    corrected_score = ((n11 - expected_overlap)/(np.minimum((n11+n01), (n11+n10)) - expected_overlap))
    corrected_score2 = ((n00*n11 - n10*n01)/((min_n01_n10+n11)*(min_n01_n10 + n00)))

    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score2, np.isnan(corrected_score2)))).all(), "Error in computing some of the index! "
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))


def corrected_positive_Jaccard(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1,bin_pred2)

    N = n00 + n11 + n10 + n01
    expected_positive_overlap = (n11+n01)*(n11+n10)/N

    corrected_score = ((n11 - expected_positive_overlap)/(n10 + n11 +n01 - expected_positive_overlap))
    corrected_score2 = (n00*n11 - n10*n01)/((n00*n11) - (n01*n10) + ((n10+n01)*N))

    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score2, np.isnan(corrected_score2)))).all(), "Error in computing some of the index! "
    # return np.ma.masked_array(corrected_score, np.isnan(corrected_score))
    return corrected_score



def corrected_IOU(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)

    N = n00 + n11 + n10 + n01
    expected_positive_overlap = (n11 + n01) * (n11 + n10) / N
    expected_negative_overlap = (n00 + n01) * (n00 + n10) / N

    corrected_score = ((n11 + n00 - expected_positive_overlap - expected_negative_overlap) /
                       (n10 + n11 + n01 + n00 - expected_positive_overlap - expected_negative_overlap))
    corrected_score2 = (2*n00 * n11 - 2*n10 * n01) / (2*(n00 * n11) - 2*(n01 * n10) + ((n10 + n01) * N))

    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score2,
                                np.isnan(corrected_score2)))).all(), "Error in computing some of the index! "
    # return np.ma.masked_array(corrected_score, np.isnan(corrected_score))
    return corrected_score


def corrected_Jaccard_pigeonhole(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)

    N = n00 + n11 + n10 + n01
    pigeonhole_positive_correction = (2*n11 + n01 + n10) - N
    max_overlap = np.maximum(pigeonhole_positive_correction, 0)

    corrected_score = ((n11 - max_overlap) /
                       (n10 + n11 + n01 - max_overlap))
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))
    # return corrected_score


def append_row_dataframe(results_to_append, threshold_binary_label, coeff_name, df ):
    # overlap_coeff = overlap_coefficient(binary_predictions1, binary_predictions2)
    row_to_append = [threshold_binary_label, coeff_name]
    row_to_append.extend(results_to_append)
    row_to_add = pd.DataFrame([row_to_append], columns=list(df.columns))
    df = df.append(row_to_add)
    return df


def make_scatterplot(y_axis_collection, y_axis_title, x_axis_collection, x_axis_title, res_path, threshold_prefix =None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    colors = cm.rainbow(np.linspace(0, 1, len(y_axis_collection)))
    for x, y, color in zip(x_axis_collection, y_axis_collection, colors):
        # x, y = pearson_corr_col, spearman_corr_col
        ax.scatter(x, y, c=color, edgecolors='none', s=30)

    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()
    if threshold_prefix is not None:
        fig.savefig(
            res_path + 'scatter_' + x_axis_title + '_' + y_axis_title + '_'+ str(threshold_prefix)+'.jpg',
            bbox_inches='tight')
    else:
        fig.savefig(res_path +  'scatter_' + x_axis_title + '_' + y_axis_title + '.jpg', bbox_inches='tight')

    plt.close(fig)


def make_scatterplot_with_errorbar(y_axis_collection, y_axis_title, x_axis_collection, x_axis_title, res_path, y_errors,
                                   error_bar = False, threshold_prefix =None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    colors = cm.rainbow(np.linspace(0, 1, len(y_axis_collection)))
    for x, y, y_error_bar, color in zip(x_axis_collection, y_axis_collection, y_errors, colors):
        # x, y = pearson_corr_col, spearman_corr_col
        ax.scatter(x, y, c=color, edgecolors='none', s=30)
        if (error_bar==True) and (y_errors is not None):
            ax.errorbar(x, y, xerr=0, yerr=y_error_bar, ecolor=color)
    # ax.set(xlim=(np.min(x), 1), ylim=(0, 1))
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()

    if threshold_prefix is not None:
        fig.savefig(
            res_path + 'scatter_' + x_axis_title + '_' + y_axis_title + '_'+ str(threshold_prefix)+'.jpg',
            bbox_inches='tight')
    else:
        fig.savefig(res_path +  'scatter_' + x_axis_title + '_' + y_axis_title + '.jpg', bbox_inches='tight')

    plt.close(fig)


# Create data
def scatterplot_AUC_stabscore(y_axis_collection1, y_axis_title1, y_axis_collection2, y_axis_title2,
                              x_axis_collection, x_axis_title, res_path, threshold):
    make_scatterplot(y_axis_collection1, y_axis_title1, x_axis_collection, x_axis_title, res_path, threshold)
    make_scatterplot(y_axis_collection2, y_axis_title2, x_axis_collection, x_axis_title, res_path, threshold)

    concat_metrics = np.append([np.asarray(y_axis_collection1)], [np.asarray(y_axis_collection2)], axis=0)
    mean_metrics = np.mean(concat_metrics, axis=0)
    stand_dev = np.std(concat_metrics, axis=0)
    # print("standard deviation ")
    # print(stand_dev)
    make_scatterplot(mean_metrics, 'mean_'+y_axis_title1 + '_'+y_axis_title2, x_axis_collection, x_axis_title,
                                   res_path, threshold_prefix=threshold)
    make_scatterplot_with_errorbar(mean_metrics, 'mean_'+y_axis_title1 + '_'+y_axis_title2 + '_error', x_axis_collection, x_axis_title,
                                   res_path, y_errors=stand_dev, error_bar=True, threshold_prefix=threshold)


def get_suffix_models(set_name1, set_name2):
    suffix_model1 = set_name1[-10:-4]
    suffix_model2 = set_name2[-10:-4]

    str_idx1 = set_name1.find('CV')
    str_idx2 = set_name2.find('CV')

    split_suffix1 = set_name1[str_idx1: (str_idx1 + 3)]
    split_suffix2 = set_name2[str_idx2: (str_idx2 + 3)]
    assert split_suffix1 == split_suffix2, "Error - you seem to compare different folds! "
    return suffix_model1, suffix_model2, split_suffix1, split_suffix2


def load_predictions(set_name1, set_name2, predict_res_path):
    patch_labels_prefix = 'patch_labels_'
    img_ind_prefix = 'image_indices_'
    raw_pred_prefix = 'predictions_'

    df_stability = pd.DataFrame()
    df_auc = pd.DataFrame()

    all_labels_1, all_image_ind_1, all_raw_predictions_1 = load_prediction_files(patch_labels_prefix, img_ind_prefix,
                                                                                 raw_pred_prefix,
                                                                                 set_name1, predict_res_path)
    all_labels_95, all_image_ind_95, all_raw_predictions_95 = load_prediction_files(patch_labels_prefix, img_ind_prefix,
                                                                                    raw_pred_prefix,
                                                                                    set_name2, predict_res_path)

    return  all_labels_1, all_image_ind_1, all_raw_predictions_1, all_labels_95, all_image_ind_95, \
            all_raw_predictions_95


def load_predictions_v2(classifier_name_list, predict_res_path):
    patch_labels_prefix = 'patch_labels_'
    img_ind_prefix = 'image_indices_'
    raw_pred_prefix = 'predictions_'

    all_labels = []
    all_image_ind = []
    all_raw_predictions = []
    for classifier in classifier_name_list:
        all_labels_classifier, all_image_ind_classifier, all_raw_predictions_classifier = load_prediction_files(patch_labels_prefix,
                                                                                        img_ind_prefix,
                                                                                        raw_pred_prefix,
                                                                                        classifier, predict_res_path)
        all_labels.append(all_labels_classifier)
        all_image_ind.append(all_image_ind_classifier)
        all_raw_predictions.append(all_raw_predictions_classifier)
    return all_labels, all_image_ind, all_raw_predictions


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


def filter_predictions_files_segmentation_images(all_labels_1, all_image_ind_1, all_raw_predictions_1, bbox_indices1,
                                                 all_labels_2, all_image_ind_2, all_raw_predictions_2, bbox_indices2):
    labels_1, image_ind_1, raw_predictions_1 = all_labels_1[bbox_indices1], all_image_ind_1[bbox_indices1], \
                                               all_raw_predictions_1[bbox_indices1]
    labels_2, image_ind_2, raw_predictions_2 = all_labels_2[bbox_indices2], all_image_ind_2[bbox_indices2], \
                                               all_raw_predictions_2[bbox_indices2]
    return  labels_1, image_ind_1, raw_predictions_1, labels_2, image_ind_2, raw_predictions_2


def filter_predictions_files_segmentation_images_v2(all_labels_coll, all_image_ind_coll, all_raw_predictions_coll,
                                                    bbox_ind_coll):
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
    assert len(bbox_ind_coll) == len(all_labels_coll) == len(all_image_ind_coll) == len(all_raw_predictions_coll), \
        "The lists do not have the same length"
    for el_ind in range(0, len(all_labels_coll)):
    # for bbox_ind in bbox_ind_coll:
        bbox_ind = bbox_ind_coll[el_ind]
        labels, image_ind, raw_predictions = (all_labels_coll[el_ind])[bbox_ind], (all_image_ind_coll[el_ind])[bbox_ind], \
                                             (all_raw_predictions_coll[el_ind])[bbox_ind]
        bbox_img_labels_coll.append(labels)
        bbox_img_ind_coll.append(image_ind)
        bbox_img_raw_predictions.append(raw_predictions)
        print("""""""""""""""""""")
        print(el_ind)
        print(image_ind)
        assert bbox_img_ind_coll[0].all() == image_ind.all(), "bbox image index are different or in different order"
    return bbox_img_labels_coll, bbox_img_ind_coll, bbox_img_raw_predictions


def prepare_dataframes_with_results(image_indices):
    df_stability = pd.DataFrame()
    df_auc = pd.DataFrame()
    df_stability['Threshold'] = None
    df_stability['Score'] = None
    df_auc['subset_model'] = None

    for image_idx in range(0, len(image_indices)):
         df_stability[(image_indices[image_idx])[-16:-4]] = None
         df_auc[(image_indices[image_idx])[-16:-4]] = None
    return df_stability, df_auc


def calculate_auc_save_in_df(raw_predictions, inst_labels, df_auc, model_suffix):
    auc_coll1 = calculate_AUC_batch(raw_predictions, inst_labels)
    print("Average instance AUC is: "+str(np.average(auc_coll1)))

    row = [model_suffix]
    row.extend(auc_coll1)
    row_to_add = pd.DataFrame([row], columns=list(df_auc.columns))
    df_auc = df_auc.append(row_to_add)
    return df_auc, auc_coll1


def get_binary_scores_forthreshold(thres, raw_pred1, raw_pred2):
    binary_predictions1 = binarize_predictions(raw_pred1, threshold=thres)
    binary_predictions2 = binarize_predictions(raw_pred2, threshold=thres)

    jaccard_indices = positive_Jaccard_index_batch(binary_predictions1, binary_predictions2, 16)
    jaccard_indices_mask = np.ma.masked_array(jaccard_indices, np.isnan(jaccard_indices))

    corrected_jacc_pigeonhole = corrected_Jaccard_pigeonhole(binary_predictions1, binary_predictions2)
    corrected_pos_jacc = corrected_positive_Jaccard(binary_predictions1, binary_predictions2)

    overlap_coeff = overlap_coefficient(binary_predictions1, binary_predictions2, 16)
    corrected_overlap = corrected_overlap_coefficient(binary_predictions1, binary_predictions2)
    corrected_iou = corrected_IOU(binary_predictions1, binary_predictions2)
    return jaccard_indices, corrected_pos_jacc, corrected_jacc_pigeonhole, overlap_coeff, corrected_overlap, corrected_iou


def get_binary_scores_forthreshold_v2(thres, raw_pred_coll):
    binary_predictions_coll = []
    jaccard_coll, corr_jacc_coll, corr_jacc_pigeonhole_coll, overlap_coll, \
    corr_overlap_coll, corr_iou_coll =[], [], [], [], [], []

    for raw_pred in raw_pred_coll:
        binary_predictions = binarize_predictions(raw_pred, threshold=thres)
        binary_predictions_coll.append(binary_predictions)

    for bin_pred in binary_predictions_coll:
        for bin_pred2 in binary_predictions_coll:

            jaccard_indices = positive_Jaccard_index_batch(bin_pred, bin_pred2, 16)
            jaccard_indices_mask = np.ma.masked_array(jaccard_indices, np.isnan(jaccard_indices))
            jaccard_coll.append(jaccard_indices)

            corrected_jacc_pigeonhole = corrected_Jaccard_pigeonhole(bin_pred, bin_pred2)
            corr_jacc_pigeonhole_coll.append(corrected_jacc_pigeonhole)

            corrected_pos_jacc = corrected_positive_Jaccard(bin_pred, bin_pred2)
            corr_jacc_coll.append(corrected_pos_jacc)

            overlap_coeff = overlap_coefficient(bin_pred, bin_pred2, 16)
            overlap_coll.append(overlap_coeff)

            corrected_overlap = corrected_overlap_coefficient(bin_pred, bin_pred2)
            corr_overlap_coll.append(corrected_overlap)

            corrected_iou = corrected_IOU(bin_pred, bin_pred2)
            corr_iou_coll.append(corrected_iou)
    return jaccard_coll, corr_jacc_coll, corr_jacc_pigeonhole_coll, overlap_coll, corr_overlap_coll, corr_iou_coll


def plot_change_stability_varying_threshold_per_image(overlap_coll, jacc_coll, corr_overlap_col, corr_jaccard_coll, corr_iou_coll,
                                            corr_jaccard_pgn_coll, img_ind, threshold_coll, res_path):
    plot_line_graph(overlap_coll, 'Overlap coefficient', jacc_coll, 'Positive Jaccard distance',
                    corr_overlap_col, 'Corrected Overlap coefficient', corr_jaccard_coll,
                    'Corrected Positive Jaccard distance', corr_iou_coll, "Corrected IoU",
                    corr_jaccard_pgn_coll, "Corrected Positive Jaccard using Pigeonhole",
                    threshold_coll, 'threshold', res_path, 'varying_thres_stability' + str(img_ind), "")


def plot_change_stability_varying_threshold(raw_predictions1, raw_predictions2, res_path, image_indices):
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    jacc_collection = []
    corr_jacc_collection = []
    corr_iou_collection = []
    jacc_pgn_collection = []
    overlap_collection = []
    corr_overlap_collection = []

    for threshold_bin in threshold_list:
        binary_predictions1 = binarize_predictions(raw_predictions1, threshold=threshold_bin)
        binary_predictions2 = binarize_predictions(raw_predictions2, threshold=threshold_bin)

        jaccard_indices = positive_Jaccard_index_batch(binary_predictions1, binary_predictions2, 16)
        jaccard_indices_mask = np.ma.masked_array(jaccard_indices, np.isnan(jaccard_indices))
        jacc_collection.append(jaccard_indices_mask)

        ############################################ Corrected Jaccard - PIGEONHOLE coefficient  #########################
        corrected_jacc_pigeonhole = corrected_Jaccard_pigeonhole(binary_predictions1, binary_predictions2)
        jacc_pgn_collection.append(corrected_jacc_pigeonhole)

        ############################################ Corrected Jaccard coefficient  #########################
        corrected_pos_jacc = corrected_positive_Jaccard(binary_predictions1, binary_predictions2)
        corr_jacc_collection.append(corrected_pos_jacc)

        ############################################  Overlap coefficient #########################
        overlap_coeff = overlap_coefficient(binary_predictions1, binary_predictions2, 16)
        overlap_collection.append(overlap_coeff)

        ############################################ Corrected overlap coefficient  #########################
        corrected_overlap = corrected_overlap_coefficient(binary_predictions1, binary_predictions2)
        corr_overlap_collection.append(corrected_overlap)

        ############################################  corrected IOU score   #########################
        corrected_iou = corrected_IOU(binary_predictions1, binary_predictions2)
        corr_iou_collection.append(corrected_iou)

    st_dev_collection = []
    for idx in range(0, len(image_indices)):
        img_index = (image_indices[idx])[-16:-4]

        plot_change_stability_varying_threshold_per_image(np.asarray(overlap_collection)[:,idx],
                                                          np.asarray(jacc_collection)[:,idx],
                                                          np.asarray(corr_overlap_collection)[:,idx],
                                                          np.asarray(corr_jacc_collection)[:,idx],
                                                          np.asarray(corr_iou_collection)[:,idx],
                                                          np.asarray(jacc_pgn_collection)[:,idx],
                                                          img_index, threshold_list, res_path)
        std = np.std(np.asarray(corr_jacc_collection)[:, idx])
        st_dev_collection.append(round(std, 4))
    print(st_dev_collection)

def compute_binary_scores_with_thresholds(raw_predictions1, raw_predictions2, df_stab, auc_1, auc_2, scatterplots, res_path):
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # jacc_collection = []
    # corr_jacc_collection =[]
    # corr_iou_collection = []
    # jacc_pgn_collection = []
    # overlap_collection = []
    # corr_overlap_collection =[]

    for threshold_bin in threshold_list:
        binary_predictions1 = binarize_predictions(raw_predictions1, threshold=threshold_bin)
        binary_predictions2 = binarize_predictions(raw_predictions2, threshold=threshold_bin)

        jaccard_indices = positive_Jaccard_index_batch(binary_predictions1, binary_predictions2, 16)
        jaccard_indices_mask = np.ma.masked_array(jaccard_indices, np.isnan(jaccard_indices))
        # # jaccard_indices_mask = np.nan_to_num(jaccard_indices)

        df_stab = append_row_dataframe(jaccard_indices_mask, threshold_bin, 'positive Jaccard', df_stab)

        ############################################ Corrected Jaccard - PIGEONHOLE coefficient  #########################
        corrected_jacc_pigeonhole = corrected_Jaccard_pigeonhole(binary_predictions1, binary_predictions2)
        df_stab = append_row_dataframe(corrected_jacc_pigeonhole, threshold_bin, 'corrected Jaccard pigeonhole',
                                            df_stab)

        ############################################ Corrected Jaccard coefficient  #########################
        corrected_pos_jacc = corrected_positive_Jaccard(binary_predictions1, binary_predictions2)
        df_stab = append_row_dataframe(corrected_pos_jacc, threshold_bin, 'corrected positive Jaccard',
                                            df_stab)

        ############################################  Overlap coefficient #########################
        overlap_coeff = overlap_coefficient(binary_predictions1, binary_predictions2, 16)
        df_stab = append_row_dataframe(np.ma.masked_array(overlap_coeff, np.isnan(overlap_coeff)),
                                            threshold_bin, 'Overlap coefficient', df_stab)

        ############################################ Corrected overlap coefficient  #########################
        corrected_overlap = corrected_overlap_coefficient(binary_predictions1, binary_predictions2)
        df_stab = append_row_dataframe(corrected_overlap, threshold_bin, 'corrected overlap coeff', df_stab)

        ############################################  corrected IOU score   #########################
        corrected_iou = corrected_IOU(binary_predictions1, binary_predictions2)
        df_stab = append_row_dataframe(corrected_iou, threshold_bin, 'corrected IoU', df_stab)

        print(abs(np.subtract(auc_1, auc_2)))
        abs_dff_auc = abs(np.subtract(auc_1, auc_2))

        if scatterplots==True:
            visualize_instAUC_vs_stability_index(auc_1, 'AUC1', auc_2, overlap_coeff, corrected_overlap, jaccard_indices,
                                                 corrected_pos_jacc, corrected_jacc_pigeonhole, corrected_iou,
                                                 res_path, threshold_bin)

            visualize_instAUC_vs_stability_index(abs_dff_auc, 'ABS_difference_AUC', auc_2, overlap_coeff, corrected_overlap,
                                                 jaccard_indices, corrected_pos_jacc, corrected_jacc_pigeonhole, corrected_iou,
                                                 res_path, threshold_bin)
        # df_stab.to_csv(
        #     stability_res_path + split_suffix1 + '_stability_index_' + suffix1 + '_' + suffix2 + '.csv')
    return df_stab


def visualize_instAUC_vs_stability_index(auc_1, auc1_text, auc_2, overlap, corr_overlap, jacc, corr_jacc, corr_jacc_pgn,
                                         corr_iou, res_path, threshold_bin):
    scatterplot_AUC_stabscore(auc_1, auc1_text, auc_2, "AUC2", overlap, "overlap_coeff", res_path,threshold_bin)
    scatterplot_AUC_stabscore(auc_1,  auc1_text, auc_2, "AUC2", corr_overlap, "corrected_overlap", res_path,
                              threshold_bin)
    scatterplot_AUC_stabscore(auc_1,  auc1_text, auc_2, "AUC2", corr_jacc_pgn, "corrected_jaccard_pigeonhole", res_path,
                              threshold_bin)
    scatterplot_AUC_stabscore(auc_1,  auc1_text, auc_2, "AUC2", corr_iou, "corrected_iou", res_path, threshold_bin)
    scatterplot_AUC_stabscore(auc_1,  auc1_text, auc_2, "AUC2", jacc, "jaccard", res_path, threshold_bin)
    scatterplot_AUC_stabscore(auc_1,  auc1_text, auc_2, "AUC2", corr_jacc, "corrected_jaccard", res_path, threshold_bin)


def compute_save_correlation_scores(raw_predictions1, raw_predictions2, df_stab, suffix1, suffix2, split_suffix1,
                                    res_path, scatterplots, auc1, auc2):
    pearson_corr_col = calculate_pearson_coefficient_batch(raw_predictions1, raw_predictions2)
    print("correlation coefficient")

    # print(pearson_corr_col[0].shape)
    print("Average correlation between two predictions is: " + np.array2string(np.average(pearson_corr_col)))

    row_corr = [0, 'Pearson_correlation']
    row_corr.extend(pearson_corr_col)
    row_to_add = pd.DataFrame([row_corr], columns=list(df_stab.columns))
    df_stab  = df_stab.append(row_to_add)


    ############################################ Spearman rank correlation coefficient  #########################
    spearman_corr_col = calculate_spearman_rank_coefficient(raw_predictions1, raw_predictions2)

    print("Average Spearman correlation between two predictions is: " + np.array2string(np.average(spearman_corr_col)))

    row_corr = [0,'Spearman rank correlation']
    row_corr.extend(spearman_corr_col)
    row_to_add = pd.DataFrame([row_corr], columns=list(df_stab.columns))
    df_stab = df_stab.append(row_to_add)

    df_stab.to_csv(
        stability_res_path + split_suffix1 + '_stability_index_' + suffix1 + '_' + suffix2 + '.csv')

    abs_dff_auc = abs(np.subtract(auc1, auc2))

    mask_high_auc = (np.array(auc1, dtype=np.float64) < 0.9) & (np.array(auc2, dtype=np.float64) < 0.9)
    high_auc_diff = np.ma.masked_where(mask_high_auc, abs_dff_auc)

    high_auc_spearman = np.ma.masked_where(mask_high_auc, spearman_corr_col)
    high_auc_pearson = np.ma.masked_where(mask_high_auc, pearson_corr_col)
    print(auc1)
    print(high_auc_spearman)


    if scatterplots==True:
        make_scatterplot(spearman_corr_col, "Spearman rank", pearson_corr_col, "Pearson", results_path)
        # make_scatterplot(auc1, "AUC1", pearson_corr_col, "Pearson", results_path)
        # make_scatterplot(auc2, "AUC2", pearson_corr_col, "Pearson", results_path)
        # make_scatterplot(auc1, "AUC1", spearman_corr_col, "Spearman rank", results_path)
        # make_scatterplot(auc2, "AUC2", spearman_corr_col, "Spearman rank", results_path)
        scatterplot_AUC_stabscore(auc1, 'AUC1', auc2, "AUC2", pearson_corr_col, "Pearson", res_path, threshold=0)
        scatterplot_AUC_stabscore(auc1, 'AUC1', auc2, "AUC2", spearman_corr_col, "Spearman", res_path, threshold=0)

        # make_scatterplot(abs_dff_auc, "ABS_difference_AUC", pearson_corr_col, "Pearson", results_path)
        # make_scatterplot(high_auc_diff, "ABS_difference_09AUC", high_auc_spearman, "Spearman rank", results_path)
        # make_scatterplot(high_auc_diff, "ABS_difference_09AUC", high_auc_pearson, "Pearson rank", results_path)

    return pearson_corr_col, spearman_corr_col


def compute_correlation_scores_v2(raw_predictions):
    pearson_corr_col = []
    spearman_corr_col = []
    for pred in raw_predictions:
        for pred2 in raw_predictions:
            pearson_corr = calculate_pearson_coefficient_batch(pred, pred2)
            spearman_corr = calculate_spearman_rank_coefficient(pred, pred2)
            pearson_corr_col.append(pearson_corr)
            spearman_corr_col.append(spearman_corr)
    return pearson_corr_col, spearman_corr_col

def visualize_single_image_1class_5classifiers(img_ind_coll,labels_coll,  raw_predictions_coll, img_path, results_path,
                                               class_name,
                                               image_title_suffix):

    for ind in range(0, img_ind_coll[0].shape[0]):
        print(ind)
        threshold_transparency = 0.01

        instance_label_gt = labels_coll[0][ind, :, :,0]
        img_ind = img_ind_coll[0][ind]
        raw_prediction = raw_predictions_coll[0][ind, :, :, 0]
        # auc = auc_score[ind]

        raw_prediction2 = raw_predictions_coll[1][ind, :, :, 0]
        # auc2 = auc_score2[ind]

        img_dir = Path(img_path + get_image_index_from_pathstring(img_ind) + '.png').__str__()
        img = plt.imread(img_dir)

        scale_width = int(img.shape[1]/16)
        scale_height =int(img.shape[0]/16)
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))


        ## SUB-GRAPH 1
        ax1 = plt.subplot(2, 3, 1)
        ax1.set_title('Predictions Classifier 1', {'fontsize': 9})
        y = (np.where(instance_label_gt == instance_label_gt.max()))[0]
        x = (np.where(instance_label_gt == instance_label_gt.max()))[1]
        bottom_x = np.amin(x)
        bottom_y = np.amax(y)
        upper_left_x = np.min(x)

        upper_left_y = np.amin(y)
        width = np.amax(x) - upper_left_x
        height = np.amax(y) - upper_left_y


        # OPENCV
        img_bbox = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_bbox, (upper_left_x * scale_width, upper_left_y * scale_height),
                                 ((np.amax(x) + 1) * scale_width, (np.amax(y) + 1) * scale_height), (125, 0, 0), 5)


        ax1.imshow(img_bbox)
        red_patch = matplotlib.patches.Patch(color='red', label='Ground truth annotation')
        plt.legend(handles=[red_patch], bbox_to_anchor=(-0.2, -0.2), loc='lower right', borderaxespad=0.)

        pred_resized = np.kron(raw_predictions_coll[0][ind, :, :, 0], np.ones((64, 64), dtype=float))
        pred_resized[pred_resized < threshold_transparency] = np.nan
        img1_mask = ax1.imshow(pred_resized, 'BuPu', zorder=0, alpha = 0.8, vmin=0, vmax=1)
        # ax1.set_xlabel("AUC instance score: "+ ("{0:.3f}".format(auc)))
        fig.colorbar(img1_mask,ax=ax1, fraction=0.046)

        fig.text(-0.2, 0.5, '\n Only patches with prediction score above '+ str(threshold_transparency) +" are shown! ",
                 horizontalalignment='center',
                 verticalalignment='center', fontsize=9)

        ## SUB-GRAPH 2
        ax2 = plt.subplot(2,3, 2)
        ax2.set_title('Predictions Classifier 2', {'fontsize': 9})

        ax2.imshow(img_bbox, 'bone')
        pred_resized2 = np.kron(raw_predictions_coll[1][ind, :, :, 0], np.ones((64, 64), dtype=float))
        pred_resized2[pred_resized2 < threshold_transparency] = np.nan
        img2_mask = ax2.imshow(pred_resized2, 'BuPu', zorder=0, alpha=0.8,  vmin=0, vmax=1)
        # ax2.set_xlabel("AUC instance score: " + ("{0:.3f}".format(auc2)))
        fig.colorbar(img2_mask, ax=ax2, fraction=0.046)

        ## SUB-GRAPH 3
        ax3 = plt.subplot(2, 3, 3)
        ax3.set_title('Predictions Classifier 3', {'fontsize': 9})

        ax3.imshow(img_bbox, 'bone')
        pred_resized3 = np.kron(raw_predictions_coll[2][ind, :, :, 0], np.ones((64, 64), dtype=float))
        pred_resized3[pred_resized3 < threshold_transparency] = np.nan
        img3_mask = ax3.imshow(pred_resized3, 'BuPu', zorder=0, alpha=0.8, vmin=0, vmax=1)
        # ax2.set_xlabel("AUC instance score: " + ("{0:.3f}".format(auc2)))
        fig.colorbar(img3_mask, ax=ax3, fraction=0.046)
        #
        ## SUB-GRAPH 4
        ax4 = plt.subplot(2, 3, 4)
        ax4.set_title('Predictions Classifier 4', {'fontsize': 9})

        ax4.imshow(img_bbox, 'bone')
        pred_resized4= np.kron(raw_predictions_coll[3][ind, :, :, 0], np.ones((64, 64), dtype=float))
        pred_resized4[pred_resized4 < threshold_transparency] = np.nan
        img4_mask = ax4.imshow(pred_resized4, 'BuPu', zorder=0, alpha=0.8, vmin=0, vmax=1)
        # ax2.set_xlabel("AUC instance score: " + ("{0:.3f}".format(auc2)))
        fig.colorbar(img4_mask, ax=ax4, fraction=0.046)


        ## SUB-GRAPH 5
        ax5 = plt.subplot(2, 3, 5)
        ax5.set_title('Predictions Classifier 5', {'fontsize': 9})

        ax5.imshow(img_bbox, 'bone')
        pred_resized5= np.kron(raw_predictions_coll[4][ind, :, :, 0], np.ones((64, 64), dtype=float))
        pred_resized5[pred_resized5 < threshold_transparency] = np.nan
        img5_mask = ax5.imshow(pred_resized5, 'BuPu', zorder=0, alpha=0.8, vmin=0, vmax=1)
        # ax2.set_xlabel("AUC instance score: " + ("{0:.3f}".format(auc2)))
        fig.colorbar(img5_mask, ax=ax5, fraction=0.046)

        # ## SUB-GRAPH 4
        # ax4 = plt.subplot(2, 3, 1)
        # ax4.set_title('Predictions Classifier 2', {'fontsize': 9})
        #
        # ax4.imshow(img_bbox, 'bone')
        # pred_resized4= np.kron(raw_predictions_coll[3][ind, :, :, 0], np.ones((64, 64), dtype=float))
        # pred_resized4[pred_resized4 < threshold_transparency] = np.nan
        # img4_mask = ax4.imshow(pred_resized4, 'BuPu', zorder=0, alpha=0.8, vmin=0, vmax=1)
        # # ax2.set_xlabel("AUC instance score: " + ("{0:.3f}".format(auc2)))
        # fig.colorbar(img4_mask, ax=ax4, fraction=0.046)


        plt.tight_layout()
        fig.savefig(results_path + get_image_index_from_pathstring(img_ind) + '_' + class_name + image_title_suffix + '.jpg',
                    bbox_inches='tight')
        plt.close(fig)

#
# # # #################################################################################################
# set_name1 = 'subset_test_set_CV0_2_0.95.npy'
# set_name2='subset_test_set_CV0_4_0.95.npy'
# suffix_1, suffix_2, split_suffix_1, split_suffix_2 = get_suffix_models(set_name1, set_name2)
#
# all_labels1, all_image_ind1, all_raw_predictions1, all_labels2, all_image_ind2,all_raw_predictions2 =\
#     load_predictions(set_name1, set_name2, prediction_results_path)
#
# bbox_indices1, bbox_indices2 = indices_segmentation_images(all_labels1, all_labels2)
#
# labels_1, image_ind_1, raw_predictions_1, labels_2, image_ind_2, raw_predictions_2 = \
#     filter_predictions_files_segmentation_images(all_labels1, all_image_ind1, all_raw_predictions1, bbox_indices1,
#                                              all_labels2, all_image_ind2,all_raw_predictions2, bbox_indices2)
# print(image_ind_1)
# df_stability, df_auc = prepare_dataframes_with_results(image_ind_1)
#
#
# df_auc, a1 = calculate_auc_save_in_df(raw_predictions_1, labels_1, df_auc, suffix_1)
#
# df_auc, a2 = calculate_auc_save_in_df(raw_predictions_2, labels_2, df_auc, suffix_2 )
#
# df_stability = compute_binary_scores_with_thresholds(raw_predictions_1, raw_predictions_2, df_stability, a1, a2,
#                                                      res_path=results_path, scatterplots=False)
#
#
# pearson_corr, spearman_rank_corr = compute_save_correlation_scores(raw_predictions_1, raw_predictions_2,df_stability,
#                                                                    suffix_1, suffix_2, split_suffix_1,
#                                                                    res_path=results_path, auc1=a1, auc2=a2,
#                                                                    scatterplots=False)
#
# jacc, corr_jacc, jacc_pigeonhole, overlap, corr_overlap, corr_iou = get_binary_scores_forthreshold(0.5, raw_predictions_1,
#                                                                                                    raw_predictions_2)
# correlation_between_scores = calculate_spearman_rank_coefficient_v2([jacc, corr_jacc, jacc_pigeonhole, corr_iou, overlap,
#                                                                      corr_overlap, pearson_corr, spearman_rank_corr])
# all_labels = ['pos Jaccard', 'Corr positive Jacc', 'Corr Jacc using Pigeonhole', 'Corr IoU', 'Overlap', 'Corr overlap',
#               'Pearson', 'Spearman rank']
# bin_labels =['pos Jaccard', 'Corr positive Jacc', 'Corr Jacc using Pigeonhole', 'Corr IoU', 'Overlap', 'Corr overlap']
# correlation_between_scores2 = calculate_spearman_rank_coefficient_v2([jacc, corr_jacc, jacc_pigeonhole, corr_iou, overlap,
#                                                                      corr_overlap])
#
# print(correlation_between_scores)
# print(correlation_between_scores2)
#
# visualize_correlation_heatmap(correlation_between_scores, results_path, "all",all_labels, dropDuplicates=True)
# visualize_correlation_heatmap(correlation_between_scores2, results_path, "binary",bin_labels, dropDuplicates=True)
#
# plot_change_stability_varying_threshold(raw_predictions_1, raw_predictions_2, results_path, image_ind_1)
