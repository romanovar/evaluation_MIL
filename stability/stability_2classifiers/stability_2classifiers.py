import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau
from scipy.stats.mstats_basic import kendalltau_seasonal
from sklearn.metrics import roc_auc_score

from stability.preprocessor.preprocessing import binarize_predictions
from stability.visualizations.visualization_utils import scatterplot_AUC_stabscore, make_scatterplot, \
    visualize_instAUC_vs_stability_index
from stability.stability_2classifiers.scores_2classifiers import positive_Jaccard_index_batch, \
    corrected_Jaccard_pigeonhole, corrected_positive_Jaccard, overlap_coefficient, corrected_IOU, \
    corrected_overlap_coefficient, calculate_spearman_rank_coefficient, calculate_pearson_coefficient_batch, \
    calculate_kendallstau_coefficient_batch


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


# def calculate_pearson_coefficient_batch(raw_pred1, raw_pred2):
#     correlation_coll = []
#     assert raw_pred1.shape == raw_pred2.shape, "Predictions don't have same shapes"
#
#     for ind in range(0, raw_pred1.shape[0]):
#         corr_coef = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
#                     raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind])
#         correlation_coll.append(corr_coef[0,1])
#         corr_coef2 = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
#                     raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind], rowvar=False)
#         assert corr_coef[0, 1]==corr_coef2[0, 1], "think on the dimensions of the correlation computed "
#
#     return correlation_coll
#

def calculate_AUC_batch(pred, labels):
    auc_coll = []
    assert pred.shape == labels.shape, "Labels and predictions not from the same shape"
    for ind in range(0, pred.shape[0]):
        auc_single_image = roc_auc_score(labels[ind, :, :, :].reshape(16*16*1), pred[ind, :, :, :].reshape(16*16*1))
        auc_coll.append(auc_single_image)
    return auc_coll



def append_row_dataframe(results_to_append, threshold_binary_label, coeff_name, df ):
    # overlap_coeff = overlap_coefficient(binary_predictions1, binary_predictions2)
    row_to_append = [threshold_binary_label, coeff_name]
    row_to_append.extend(results_to_append)
    row_to_add = pd.DataFrame([row_to_append], columns=list(df.columns))
    df = df.append(row_to_add)
    return df


def get_suffix_models(set_name1, set_name2):
    suffix_model1 = set_name1[-10:-4]
    suffix_model2 = set_name2[-10:-4]

    str_idx1 = set_name1.find('CV')
    str_idx2 = set_name2.find('CV')

    split_suffix1 = set_name1[str_idx1: (str_idx1 + 3)]
    split_suffix2 = set_name2[str_idx2: (str_idx2 + 3)]
    assert split_suffix1 == split_suffix2, "Error - you seem to compare different folds! "
    return suffix_model1, suffix_model2, split_suffix1, split_suffix2


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
            # jaccard_indices_mask = np.ma.masked_array(jaccard_indices, np.isnan(jaccard_indices))
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


def compute_binary_scores_with_allthresholds(raw_predictions1, raw_predictions2, df_stab, auc_1, auc_2, scatterplots,
                                             res_path):
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
        res_path + split_suffix1 + '_stability_index_' + suffix1 + '_' + suffix2 + '.csv')

    abs_dff_auc = abs(np.subtract(auc1, auc2))

    mask_high_auc = (np.array(auc1, dtype=np.float64) < 0.9) & (np.array(auc2, dtype=np.float64) < 0.9)
    high_auc_diff = np.ma.masked_where(mask_high_auc, abs_dff_auc)

    high_auc_spearman = np.ma.masked_where(mask_high_auc, spearman_corr_col)
    high_auc_pearson = np.ma.masked_where(mask_high_auc, pearson_corr_col)
    print(auc1)
    print(high_auc_spearman)

    tau_coll= calculate_kendallstau_coefficient_batch(raw_predictions1, raw_predictions2)

    if scatterplots==True:
        make_scatterplot(spearman_corr_col, "Spearman rank", pearson_corr_col, "Pearson", res_path)
        # make_scatterplot(auc1, "AUC1", pearson_corr_col, "Pearson", results_path)
        # make_scatterplot(auc2, "AUC2", pearson_corr_col, "Pearson", results_path)
        # make_scatterplot(auc1, "AUC1", spearman_corr_col, "Spearman rank", results_path)
        # make_scatterplot(auc2, "AUC2", spearman_corr_col, "Spearman rank", results_path)
        scatterplot_AUC_stabscore(auc1, 'AUC1', auc2, "AUC2", pearson_corr_col, "Pearson", res_path, threshold=0)
        scatterplot_AUC_stabscore(auc1, 'AUC1', auc2, "AUC2", spearman_corr_col, "Spearman", res_path, threshold=0)
        scatterplot_AUC_stabscore(auc1, 'AUC1', auc2, "AUC2", tau_coll, "kendalltau", res_path, threshold=0)

        # make_scatterplot(abs_dff_auc, "ABS_difference_AUC", pearson_corr_col, "Pearson", results_path)
        # make_scatterplot(high_auc_diff, "ABS_difference_09AUC", high_auc_spearman, "Spearman rank", results_path)
        # make_scatterplot(high_auc_diff, "ABS_difference_09AUC", high_auc_pearson, "Pearson rank", results_path)

    return pearson_corr_col, spearman_corr_col, tau_coll


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
# df_stability = compute_binary_scores_with_allthresholds(raw_predictions_1, raw_predictions_2, df_stability, a1, a2,
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
