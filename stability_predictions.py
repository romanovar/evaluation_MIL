import yaml
import argparse
import os
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
import pandas as pd

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


def positive_Jaccard_index_batch(bin_pred1, bin_pred2):
    """

    :param bin_pred1: raw predictions of all bbox images
    :param bin_pred2: raw predictions of another subset of all bbox images
    :return: array with positive  jaccard index for
    """
    sum_preds = bin_pred1 + bin_pred2
    n11_mask = np.array(sum_preds > 1, dtype=int)
    n10_n01_mask = np.array(sum_preds ==1, dtype=int)

    n11 = np.sum(n11_mask.reshape((n11_mask.shape[0], 16*16*1)), axis=1)
    n10_n01 = np.sum(n10_n01_mask.reshape((n10_n01_mask.shape[0], 16*16*1)), axis=1)
    print(n11)
    print(n10_n01)

    np.seterr(divide='ignore', invalid='ignore')
    pos_jaccard_dist = n11/(n11+n10_n01)
    return pos_jaccard_dist

def calculate_spearman_rank_coefficient(pred1, pred2):
    rank_coll1 = []
    rank_coll2 = []
    spearman_corr_coll = []
    assert pred1.shape[0] == pred2.shape[0], "Ensure the predictions have same shape!"
    for obs in range(0, pred1.shape[0]):
        rank_image1 = rankdata(pred1.reshape(-1, 16*16*1)[obs])
        rank_image2 = rankdata(pred2.reshape(-1, 16*16*1)[obs])
        rho, pval = spearmanr(rank_image1, rank_image2)
        spearman_corr_coll.append(rho)
    return spearman_corr_coll

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


def corrected_index(bin_pred1, bin_pred2):
    sum_preds = bin_pred1 + bin_pred2
    n11_mask = np.array(sum_preds > 1, dtype=int)
    n00_mask = np.array(sum_preds == 0, dtype=int)

    # REMOVES ELEMENTS EQUAL 0, SO ONLY 1 are left
    pred1_n1_mask2 = np.ma.masked_equal(bin_pred1, 0)
    pred1_n0_mask2 = np.ma.masked_equal(bin_pred1, 1)

    pred2_n1_mask2 = np.ma.masked_equal(bin_pred2, 0)
    pred2_n0_mask2 = np.ma.masked_equal(bin_pred2, 1)

    n10_2 = np.sum((pred1_n1_mask2 + pred2_n0_mask2).reshape(-1, 16 * 16 * 1), axis=1)
    n01_2 = np.sum((pred1_n0_mask2 + pred2_n1_mask2).reshape(-1, 16 * 16 * 1), axis=1)
    n10 = np.asarray(n10_2)
    n01 = np.asarray(n01_2)

    # SWAPS 1s with 0s and 0s with 1s
    # pred1_n0_mask = np.array(bin_pred1==0, dtype=int)
    # pred2_n0_mask = np.array(bin_pred2==0, dtype=int)
    #
    # n10 = np.array((bin_pred1+pred2_n0_mask).reshape(-1, 16*16*1)==1, dtype=int)
    # n10 = np.sum(n10, axis=1)
    # n01=np.sum((pred1_n0_mask+bin_pred2).reshape(-1, 16*16*1), axis=1)

    n11 = np.sum(n11_mask.reshape((n11_mask.shape[0], 16 * 16 * 1)), axis=1)
    n00 = np.sum(n00_mask.reshape((n00_mask.shape[0], 16 * 16 * 1)), axis=1)

    min_n01_n10 = np.minimum(n10, n01)
    # assert (n11+n01) == np.sum(np.ma.masked_equal(bin_pred2, 0).reshape(-1, 16*16*1), axis=1),\
    #     "Error with computing the positive instances "
    # assert (n11+n10) == np.sum(np.ma.masked_equal(bin_pred1, 0).reshape(-1, 16*16*1), axis=1),\
    #     "Error with computing the positive instances "
    N = n00 + n11 + n10 + n01
    expected_overlap = (n11+n01)*(n11+n10)/N
    # 0/0 -> nan so convert nan to 0
    corrected_score = np.nan_to_num((n11 - expected_overlap)/(np.minimum((n11+n01), (n11+n10)) - expected_overlap))
    corrected_score2 = np.nan_to_num((n00*n11 - n10*n01)/((min_n01_n10+n11)*(min_n01_n10 + n00)))
    assert (corrected_score==corrected_score2).all(), "Error in computing some of the index! "
    return corrected_score

# #################################################################################################

set_name1 = 'Cardiomegalytest_set_CV2_1.00.npy'
set_name2 = 'Cardiomegaly_test_set_CV2_subset_0.95.npy'
suffix_model1 = set_name1[-8:-4]
suffix_model2 = set_name2[-8:-4]

str_idx1 = set_name1.find('CV')
str_idx2 = set_name2.find('CV')
print(suffix_model1)
print(suffix_model2)
split_suffix1 = set_name1[str_idx1: (str_idx1 + 3)]
split_suffix2 = set_name2[str_idx2: (str_idx2 + 3)]
assert split_suffix1==split_suffix2, "Error - you seem to compare different folds! "


patch_labels_prefix = 'patch_labels_'
img_ind_prefix = 'image_indices_'
raw_pred_prefix = 'predictions_'

df_stability = pd.DataFrame()
df_auc = pd.DataFrame()

all_labels_1, all_image_ind_1, all_raw_predictions_1 = load_prediction_files(patch_labels_prefix, img_ind_prefix, raw_pred_prefix,
                                                                 set_name1, prediction_results_path)
all_labels_95, all_image_ind_95, all_raw_predictions_95 = load_prediction_files(patch_labels_prefix, img_ind_prefix, raw_pred_prefix,
                                                                 set_name2, prediction_results_path)
all_image_ind_95[14]
######################################## FILTER BBOX IMAGES ######################################

bbox_indices1 =  filter_bbox_image_ind(all_labels_1)
bbox_indices95 = filter_bbox_image_ind(all_labels_95)
assert bbox_indices1==bbox_indices95, "Error, bbox images should be equal " \
                                                                                "in both cases"

print("Total images found with segmenation is: "+ str(len(bbox_indices95)))
print(len(bbox_indices1))
print(all_image_ind_95[bbox_indices1])

labels_1, image_ind_1, raw_predictions_1 = all_labels_1[bbox_indices1], all_image_ind_1[bbox_indices1], \
                                           all_raw_predictions_1[bbox_indices1]
labels_95, image_ind_95, raw_predictions_95 = all_labels_95[bbox_indices95], all_image_ind_95[bbox_indices95], \
                                           all_raw_predictions_95[bbox_indices95]

df_stability['Threshold'] = None
df_stability['Score'] = None
df_auc['subset_model'] = None
for image_idx in range(0, len(image_ind_1)):
    df_stability[(image_ind_1[image_idx])[-16:-4]] = None
    df_auc[(image_ind_1[image_idx])[-16:-4]] = None
############################################ Positive Jacard index #########################
for threshold_bin in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    binary_predictions1 = binarize_predictions(raw_predictions_1, threshold=threshold_bin)
    binary_predictions95 = binarize_predictions(raw_predictions_95, threshold=threshold_bin)
    jaccard_indices = positive_Jaccard_index_batch(binary_predictions1, binary_predictions95)
    # jaccard_indices_mask = np.ma.masked_array(jaccard_indices, np.isnan(jaccard_indices))
    jaccard_indices_mask = np.nan_to_num(jaccard_indices)
    # print(jaccard_indices_mask)
    # print(np.sum(jaccard_indices>0.7, dtype=int))
    print("Average jaccard index is: "+str(np.average(jaccard_indices_mask)))
    print(str(threshold_bin)+" threshold, jaccard index is: "+np.array2string(jaccard_indices_mask))

    # np.save(stability_res_path+'positivejaccard_thres'+str(threshold_bin) + '_' +str(suffix_model1)+'_'+str(suffix_model2), jaccard_indices_mask )
    # np.savetxt(stability_res_path+"possjacc.csv", jaccard_indices_mask, delimiter=",")
    row = [threshold_bin, 'Jaccard']
    row.extend(jaccard_indices_mask)
    row_to_add = pd.DataFrame([row], columns=list(df_stability.columns))
    df_stability = df_stability.append(row_to_add)

    ############################################ Corrected score coefficient  #########################
    corrected_scores = corrected_index(binary_predictions1, binary_predictions95)
    print("Average corrected score between two predictions is: "+str(np.average(corrected_scores)))
    print(str(threshold_bin)+" threshold, corrected score between two predictions is: "+np.array2string(corrected_scores))
    print(np.array2string(corrected_scores, separator=', '))
    # np.save(stability_res_path+'expectedcorrection_'+str(threshold_bin) + '_'+str(suffix_model1)+'_'+str(suffix_model2), corrected_scores )
    row2 = [threshold_bin, 'Expectation_correction']
    row2.extend(corrected_scores)
    row_to_add = pd.DataFrame([row2], columns=list(df_stability.columns))
    df_stability = df_stability.append(row_to_add)

    ############################################ IOU score   #########################
    iou_scores = calculate_IoU(binary_predictions1, binary_predictions95)

    row3 = [threshold_bin, 'IoU']
    row3.extend(iou_scores)
    row_to_add = pd.DataFrame([row3], columns=list(df_stability.columns))
    df_stability = df_stability.append(row_to_add)

############################################ Pearson correlation coefficient  #########################
print(raw_predictions_95.shape)
corr_col = calculate_pearson_coefficient_batch(raw_predictions_1, raw_predictions_95)
print("correlation coefficient")
print(corr_col[0].shape)
print("Average correlation between two predictions is: "+np.array2string(np.average(corr_col)))
# np.save(stability_res_path+'pearsoncorr_'+str(suffix_model1)+'_'+str(suffix_model2), corr_col )
# df_stability['correlation_coeff'] = corr_col
row_corr = [0, 'Pearson_correlation']
row_corr.extend(corr_col)
row_to_add = pd.DataFrame([row_corr], columns=list(df_stability.columns))
df_stability = df_stability.append(row_to_add)

# df_stability.to_csv(stability_res_path + split_suffix1+ '_stability_index'+ suffix_model1+'_'+suffix_model2 + '.csv')

############################################ Spearman rank correlation coefficient  #########################
print(raw_predictions_95.shape)
spearman_corr_col = calculate_spearman_rank_coefficient(raw_predictions_1, raw_predictions_95)

print("Average Spearman correlation between two predictions is: "+np.array2string(np.average(spearman_corr_col)))
# np.save(stability_res_path+'pearsoncorr_'+str(suffix_model1)+'_'+str(suffix_model2), corr_col )
# df_stability['correlation_coeff'] = corr_col
row_corr = [0, 'Spearman rank correlation']
row_corr.extend(spearman_corr_col)
row_to_add = pd.DataFrame([row_corr], columns=list(df_stability.columns))
df_stability = df_stability.append(row_to_add)

df_stability.to_csv(stability_res_path + split_suffix1+ '_stability_index'+ suffix_model1+'_'+suffix_model2 + '.csv')



############################################ Instance AUC  #########################
# df_auc = pd.DataFrame()
auc_coll1 = calculate_AUC_batch(raw_predictions_1, labels_1)
print("Average instance AUC is: "+str(np.average(auc_coll1)))

row = [suffix_model1]
row.extend(auc_coll1)
row_to_add = pd.DataFrame([row], columns=list(df_auc.columns))
df_auc = df_auc.append(row_to_add)

auc_coll95 = calculate_AUC_batch(raw_predictions_95, labels_95)
print("Average instance AUC is: "+str(np.average(auc_coll95)))

row2 = [suffix_model2]
row2.extend(auc_coll95)
row_to_add2 = pd.DataFrame([row2], columns=list(df_auc.columns))
df_auc = df_auc.append(row_to_add2)
df_auc.to_csv(stability_res_path + split_suffix1+ '_inst_auc'+ '.csv')

