import numpy as np
from scipy.stats import rankdata, spearmanr, kendalltau
from stability.preprocessing import calculate_subsets_between_two_classifiers, binarize_predictions


def calculate_positive_Jaccard(bin_pred1, bin_pred2, P):
    """
    Calculates the positive Jaccard index between predictions of two models.
    Positive Jaccard index compares **binary** predictions.
    The positive Jaccard is computed for each image.
    FORMULA: Positive Jaccard = n11/ (n01 + n10 + n11), where
                                n11 are the number of patches on an image that are predicted as positive by both models
                                n01 are the number of patches on an image that are predicted as
                                        negative by the first model and as positive by the second model
                                n10 are the number of patches on an image that are predicted as
                                        positive by the first model and as negative by the second model
    :param bin_pred1: binary predictions of a model
    :param bin_pred2:  binary predictions of another model on the same samples
    :param P: patch sizes of an image
    :return: A list of positive Jaccard index, where each element is the index of a sample/image.
    """
    sum_preds = bin_pred1 + bin_pred2
    n11_mask = np.array(sum_preds > 1, dtype=int)
    n10_n01_mask = np.array(sum_preds ==1, dtype=int)

    n11 = np.sum(n11_mask.reshape((n11_mask.shape[0], P*P)), axis=1)
    n10_n01 = np.sum(n10_n01_mask.reshape((n10_n01_mask.shape[0], P*P)), axis=1)
    np.seterr(divide='ignore', invalid='ignore')
    pos_jaccard_dist = n11/(n11+n10_n01)

    # Test if the correlation is correct

    # n00, n10, n01, n11
    d, c, b, a  = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)
    # N = a + d  + b + c
    # expected_positive_overlap = (n11 + n01) * (n11 + n10) / N
    corrected_score2 = a / (a + b + c)

    assert ((np.ma.masked_array(pos_jaccard_dist, np.isnan(pos_jaccard_dist)) ==
             np.ma.masked_array(corrected_score2,
                                np.isnan(corrected_score2)))).all(), \
        "Error in computing some of the index! Or the stability scores for all images are NaNs - this is possible if all" \
        "predictions of both models belong to the same class (all predictions are positive or negative according to" \
        " both models)!  Please, check the code"
    return pos_jaccard_dist


def calculate_spearman_rank_coefficient(raw_pred1, raw_pred2):
    """
    Calculating the Spearman rank correlation coefficient between predictions of two models.
    First the raw predictions are ranked and then the coefficient calculates the correlation between two predictions,
     based on the ranking of the patches within a sample/image.
    It is important that the predictions are done on the same samples, and the samples are in the same order in the two
    predictions.

    :param raw_pred1: predictions of a model
    :param raw_pred2:  predictions of another model on the same samples
    :return: A list with Spearman rank correlation coefficient between predictions of two models. Each element of the
    list is the correlation coefficient for a single sample/image.
    """
    spearman_corr_coll = []
    assert raw_pred1.shape[0] == raw_pred2.shape[0], "Ensure the predictions have same shape!"
    for obs in range(0, raw_pred1.shape[0]):
        rank_image1 = rankdata(raw_pred1.reshape(-1, 16 * 16 * 1)[obs])
        rank_image2 = rankdata(raw_pred2.reshape(-1, 16 * 16 * 1)[obs])
        rho, pval = spearmanr(rank_image1, rank_image2)
        spearman_corr_coll.append(rho)
    return spearman_corr_coll

# todo: delete because it is not used
# def calculate_IoU(bin_pred1, bin_pred2):
#     sum_preds = bin_pred1 + bin_pred2
#     n11_mask = np.array(sum_preds > 1, dtype=int)
#     n00_mask = np.array(sum_preds == 0, dtype=int)
#     n10_n01_mask = np.array(sum_preds == 1, dtype=int)
#
#     n11 = np.sum(n11_mask.reshape((n11_mask.shape[0], 16 * 16 * 1)), axis=1)
#     n00 = np.sum(n00_mask.reshape((n00_mask.shape[0], 16*16*1)), axis=1)
#     n10_n01 = np.sum(n10_n01_mask.reshape((n10_n01_mask.shape[0], 16 * 16 * 1)), axis=1)
#     np.seterr(divide='ignore', invalid='ignore')
#
#     iou_score = (n11+n00) / (n11 + n10_n01+n00)
#     return iou_score


def calculate_pearson_coefficient(raw_pred1, raw_pred2):
    """
    Calculating the Pearson's correlation coefficient between predictions of two models.
    It is important that the predictions are done on the same samples, and the samples are in the same order in the two
    predictions.

    :param raw_pred1: predictions of a model
    :param raw_pred2:  predictions of another model on the same samples
    :return: A list with Pearson's correlation coefficient between predictions of two models. Each element of the list
    is the correlation coefficient for a single sample/image.
    """
    correlation_coll = []
    assert raw_pred1.shape == raw_pred2.shape, "Predictions don't have same shapes, you don't compare the same samples!"

    for ind in range(0, raw_pred1.shape[0]):
        corr_coef = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
                    raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind])
        correlation_coll.append(corr_coef[0,1])
        # # Test if the correlation is correct
        # corr_coef_test = np.corrcoef(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
        #             raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind], rowvar=False)
        # assert corr_coef[0, 1] == corr_coef_test[0, 1], "think on the dimensions of the correlation computed "

    return correlation_coll


def calculate_kendallstau_coefficient_batch(raw_pred1, raw_pred2):
    correlation_coll = []
    assert raw_pred1.shape == raw_pred2.shape, "Predictions don't have same shapes"

    for ind in range(0, raw_pred1.shape[0]):
        corr_coef  = kendalltau(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
                    raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind])
        correlation_coll.append(corr_coef[0])
        corr_coef2 = kendalltau(raw_pred1.reshape((raw_pred1.shape[0], 16*16*1))[ind],
                    raw_pred2.reshape((raw_pred2.shape[0], 16*16*1))[ind])
        assert corr_coef[0]==corr_coef2[0], "think on the dimensions of the correlation computed "

    return correlation_coll


def calculate_corrected_IOU(bin_pred1, bin_pred2):
    """
    Calculates the corrected IOU index between predictions of two models.
    The index computes the intersection over union (IOU) index and corrects for overlap per chance of positive and
    negative patches.
    The index compares **binary** predictions and it is computed for each image.
    FORMULA: Positive Correction = (n10 + n11)*(n01+n11)/N, where
                                n11 are the number of patches on an image that are predicted as positive by both models
                                n01 are the number of patches on an image that are predicted as
                                        negative by the first model and as positive by the second model
                                n10 are the number of patches on an image that are predicted as
                                        positive by the first model and as negative by the second model
                                n00 are the number of patches on an image that are predicted as negative by
                                                                                                both models
                                N = n01 + n11 + n10 + n00 is the sum of all patches on an image
            Negative Correction = (n00 + n01)(n10 + n00)/N
    Corected IOU = (n11 + n00 - Positive Correction - Negative Correction)/(N - Positive Correction-Negative Correction)
    :param bin_pred1: binary predictions of a model
    :param bin_pred2:  binary predictions of another model on the same samples
    :param P: patch sizes of an image
    :return: A list of positive Jaccard index, where each element is the index of a sample/image.

    """
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)

    N = n00 + n11 + n10 + n01
    expected_positive_overlap = (((n11 + n01)/N) * ((n11 + n10)/N))* N
    expected_negative_overlap = (((n00 + n01)/N) * ((n00 + n10)/N)) * N

    corrected_score = ((n11 + n00 - expected_positive_overlap - expected_negative_overlap) /
                       (n10 + n11 + n01 + n00 - expected_positive_overlap - expected_negative_overlap))

    # # Test if the index is correct
    corrected_score2 = (2*n00 * n11 - 2*n10 * n01) / (2*(n00 * n11) - 2*(n01 * n10) + ((n10 + n01) * N))
    simplf_div = (n11*n10 + n11*n01 + 2*n11*n00 + n10*n10 + n10*n00 + n01*n01 + n01*n00)
    corrected_score3 = (2*n00 * n11 - 2*n10 * n01) /(simplf_div)
    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score2,
                                np.isnan(corrected_score2)))).all(), \
        "Error in computing some of the index! Or the stability scores for all images are NaNs - this is possible if " \
        "all" \
        "predictions of both models belong to the same class (all predictions are positive or negative according to" \
        " both models)!  Please, check the code"
    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score3,
                                np.isnan(corrected_score3)))).all(),\
        "Error in computing some of the index! Or the stability scores for all images are NaNs - this is possible if " \
        "all" \
        "predictions of both models belong to the same class (all predictions are positive or negative according to" \
        " both models)!  Please, check the code"
    return corrected_score


#### SOURCE: "High agreement but low kappa: II. Resolving the paradoxes" Cicchetti, Feinstein
def compute_positive_agreement_ratio(positive_overlap, observer1_positive, observer2_positive):
    return (2*positive_overlap)/(observer1_positive + observer2_positive)


def compute_negative_agreement_ratio(negative_overlap, observer1_negative, observer2_negative):
    return (2*negative_overlap)/(observer1_negative + observer2_negative)


def compute_total_agreement_ratio(neg_overlap, pos_overlap, N):
    return (pos_overlap+neg_overlap)/N


def compute_f1_f2(observer1_pos, observer1_neg):
    return observer1_pos - observer1_neg


def compute_g1_g2(observer2_pos, observer2_neg):
    return observer2_pos - observer2_neg


def compute_additional_scores_kappa(bin_pred1, bin_pred2):
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1,bin_pred2)
    observer1_positive = n11 + n10
    observer2_positive = n11 + n01
    p_pos = compute_positive_agreement_ratio(n11, observer1_positive, observer2_positive)
    p_neg = compute_negative_agreement_ratio(n00, observer1_negative=n00+n01, observer2_negative=n00+n10)
    po= compute_total_agreement_ratio(n00, n11, n11+n00+n10+n01)
    f1_f2 = compute_f1_f2(n11+n10, n00+n01)
    g1_g2 = compute_g1_g2(n11 + n01, n00 + n10)
    return po, p_pos, p_neg, p_pos-p_neg, f1_f2, g1_g2


def calculate_corrected_positive_Jaccard(bin_pred1, bin_pred2):
    """
    Calculates the positive Jaccard index with correction for chance. The score is between predictions of two models.
    Corrected positive Jaccard index compares **binary** predictions.
    The Corrected positive Jaccard is computed for each image.
    FORMULA: Correction = ((n10+n11) * (n01+n11))/N
                                        n11 are the number of patches on an image that are predicted as positive by
                                                                                                        both models
                                        n01 are the number of patches on an image that are predicted as
                                                negative by the first model and as positive by the second model
                                        n10 are the number of patches on an image that are predicted as
                                                positive by the first model and as negative by the second model
                                        n00 are the number of patches on an image that are predicted as negative by
                                                                                                        both models
                                        N = n01 + n11 + n10 + n00 is the sum of all patches on an image
      corrected positive Jaccard = (n11 - correction)/ (n01 + n10 + n11 - correction)


    :param bin_pred1: binary predictions of a model
    :param bin_pred2: binary predictions of another model on the same samples
    :return: A list of corrected positive Jaccard index, where each element is the index corresponding for a separate
    sample/image.
    """
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1,bin_pred2)

    N = n00 + n11 + n10 + n01
    expected_positive_overlap = (n11+n01)*(n11+n10)/N

    corrected_score = ((n11 - expected_positive_overlap)/(n10 + n11 +n01 - expected_positive_overlap))
    # Test if the score is correct
    corrected_score2 = (n00*n11 - n10*n01)/((n00*n11) - (n01*n10) + ((n10+n01)*N))
    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score2, np.isnan(corrected_score2)))).all(), \
        "Error in computing some of the index! Or the stability scores for all images are NaNs - this is possible if all" \
        "predictions of both models belong to the same class (all predictions are positive or negative accroding to" \
        " both models)!  Please, check the code"
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))


def calculate_corrected_Jaccard_heuristic(bin_pred1, bin_pred2):
    """
    Calculates the Jaccard index with heuristic correction between predictions of two models.
    Positive Jaccard index with heuristic correction compares **binary** predictions.
    The positive Jaccard with heuristic correction is computed for each image.
    FORMULA: Heuristic_correction = max(n01 +n11 + n10 + n11 - N, 0)
                                        n11 are the number of patches on an image that are predicted as positive by
                                                                                                        both models
                                        n01 are the number of patches on an image that are predicted as
                                                negative by the first model and as positive by the second model
                                        n10 are the number of patches on an image that are predicted as
                                                positive by the first model and as negative by the second model
                                        n00 are the number of patches on an image that are predicted as negative by
                                                                                                        both models
                                        N = n01 + n11 + n10 + n00 is the sum of all patches on an image
     positive Jaccard with heuristic correction = (n11 - heuristic_correction)/ (n01 + n10 + n11 -heuristic_correction)


    :param bin_pred1: binary predictions of a model
    :param bin_pred2: binary predictions of another model on the same samples
    :return: A list of positive Jaccard index with heuristic correction, where each element is the index corresponding
    for a separate sample/image.
    """
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2)

    N = n00 + n11 + n10 + n01
    pigeonhole_positive_correction = (2*n11 + n01 + n10) - N
    max_overlap = np.maximum(pigeonhole_positive_correction, 0)

    corrected_score = ((n11 - max_overlap) /
                       (n10 + n11 + n01 - max_overlap))
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))


def calculate_positive_overlap(bin_pred1, bin_pred2, P):
    """
    Calculates the overlap index on positive patches between predictions of two models.
    Ovarlap index compares **binary** predictions and it is computed for each image.
        FORMULA: Positive overlap = n11/( min(n01, n10) + n11)
    :param bin_pred1:binary predictions of a model
    :param bin_pred2: binary predictions of another model on the same samples
    :param P: the number of patches an image is divided into
    :return: A list of positive overlap index, where each element is the index corresponding for a separate sample/image

    """
    n00, n10, n01, n11 = calculate_subsets_between_two_classifiers(bin_pred1, bin_pred2, P)
    min_n01_n10 = np.minimum(n10, n01)
    return n11/(min_n01_n10 + n11)


def calculate_corrected_positive_overlap(bin_pred1, bin_pred2):
    """
    Calculates the corrected positive overlap index between predictions of two models.
    The index computes the overlap index but only on patches predicted by both models as positive.
     Additionally, the index corrects for overlap of postive patches per chance.
    Corrected positive overlap index compares **binary** predictions.
    The index is computed for each image.
        Correction = ((n10+n11) * (n01+n11))/N
                                        n11 are the number of patches on an image that are predicted as positive by
                                                                                                        both models
                                        n01 are the number of patches on an image that are predicted as
                                                negative by the first model and as positive by the second model
                                        n10 are the number of patches on an image that are predicted as
                                                positive by the first model and as negative by the second model
                                        n00 are the number of patches on an image that are predicted as negative by
                                                                                                        both models
                                        N = n01 + n11 + n10 + n00 is the sum of all patches on an image
        FORMULA: Corrected positive overlap = n11 - Correction /( min(n01, n10) + n11 - Correction)
    :param bin_pred1::binary predictions of a model
    :param bin_pred2: binary predictions of another model on the same samples
    :return: A list of corrected positive overlap indices, where each element is the index corresponding for a separate
     sample/image.
    """
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
    corrected_score = ((n11 - expected_overlap)/(np.minimum((n11+n01), (n11+n10)) - expected_overlap))

    # # Test if the index is correct
    # corrected_score2 = np.nan_to_num((n00*n11 - n10*n01)/((min_n01_n10+n11)*(min_n01_n10 + n00)))
    corrected_score2 = ((n00*n11 - n10*n01)/((min_n01_n10+n11)*(min_n01_n10 + n00)))

    assert ((np.ma.masked_array(corrected_score, np.isnan(corrected_score)) ==
             np.ma.masked_array(corrected_score2, np.isnan(corrected_score2)))).all(), \
        "Error in computing some of the index! Or the stability scores for all images are NaNs - this is possible if all" \
        "predictions of both models belong to the same class (all predictions are positive or negative accroding to" \
        " both models)!  Please, check the code"
    return np.ma.masked_array(corrected_score, np.isnan(corrected_score))


def compute_continuous_stability_scores(raw_predictions):
    """
    Computes the stability scores that use continuous [0, 1] predictions.
    A stability is always a score derived from pairwise comparison of two predictions on the same image.

    Here we just compute the stability of all possible prediction pairs. The results contain to duplicates
        (e.g. predictions of Model #1 with predictions of Model #2
        and predictions of Model #2 with predictions of Model #1)
    Results contain comparisons with itself (e.g. prediction of Model#1 with predictions of Model #1)
    :param raw_predictions: Raw predictions which are NOT binary (0/1)
    :return: List of Peason's rank correlation coefficient and Spearman's rho correlation between all prediction pairs
    """
    pearson_corr_col = []
    spearman_corr_col = []
    for pred_inner in raw_predictions:
        for pred_outer in raw_predictions:
            pearson_corr = calculate_pearson_coefficient(pred_inner, pred_outer)
            spearman_corr = calculate_spearman_rank_coefficient(pred_inner, pred_outer)
            pearson_corr_col.append(pearson_corr)
            spearman_corr_col.append(spearman_corr)
    return pearson_corr_col, spearman_corr_col


def compute_binary_stability_scores(threshold, raw_pred_coll):
    """
    Computes the stability scores that use binary (0/1) predictions after converting the raw predictions to binary.
    A stability is always a score derived from pairwise comparison of two predictions on the same image.

    Here we just compute the stability of all possible prediction pairs. The results contain to duplicates
        (e.g. predictions of Model #1 with predictions of Model #2
        and predictions of Model #2 with predictions of Model #1)
    Results contain comparisons with itself (e.g. prediction of Model#1 with predictions of Model #1)

    :param threshold: threshold for binarization
    :param raw_pred_coll: raw predictions
    :return: List of positive Jaccard, corrected positive Jaccard, heuristic correction of positive jaccard , overlap,
     positive overlap  and corrected IOU(Jaccard) from each pairwise comparison
    """
    binary_predictions_coll = []
    jaccard_coll, corr_jacc_coll, heur_corr_jacc_coll, overlap_coll, \
        corr_overlap_coll, corr_iou_coll = [], [], [], [], [], []

    for raw_pred in raw_pred_coll:
        binary_predictions = binarize_predictions(raw_pred, threshold=threshold)
        binary_predictions_coll.append(binary_predictions)

    for bin_pred_outer in binary_predictions_coll:
        for bin_pred_inner in binary_predictions_coll:
            jaccard_indices = calculate_positive_Jaccard(bin_pred_outer, bin_pred_inner, 16)
            jaccard_coll.append(jaccard_indices)

            heur_corrected_jacc = calculate_corrected_Jaccard_heuristic(bin_pred_outer, bin_pred_inner)
            heur_corr_jacc_coll.append(heur_corrected_jacc)

            corrected_pos_jacc = calculate_corrected_positive_Jaccard(bin_pred_outer, bin_pred_inner)
            corr_jacc_coll.append(corrected_pos_jacc)

            overlap_coeff = calculate_positive_overlap(bin_pred_outer, bin_pred_inner, 16)
            overlap_coll.append(overlap_coeff)

            corrected_overlap = calculate_corrected_positive_overlap(bin_pred_outer, bin_pred_inner)
            corr_overlap_coll.append(corrected_overlap)

            corrected_iou = calculate_corrected_IOU(bin_pred_outer, bin_pred_inner)
            corr_iou_coll.append(corrected_iou)
    return jaccard_coll, corr_jacc_coll, heur_corr_jacc_coll, overlap_coll, corr_overlap_coll, corr_iou_coll


def compute_stability_scores(raw_predictions_collection, bin_threshold=0.5):
    '''
    Computes the stability scores between models. For models considering binary predictions (0/1 predictions),
    a threshold of 0.5 is used for the binarization of the raw predictions
    :param raw_predictions_collection: a collection where each element is a list with the raw predictions of a models
    :param bin_threshold: a threshold used for the binarization of raw predictions to binary ones.
                            Binary predictions are needed for some of the  stability scores.
    :return: Computes all stability scores - positive Jaccard, Corrected positive Jaccard, positive Jaccard with
    heuristic correction, positive overlap, corrected positive overlap, corrected IOU
    '''

    pos_jacc, corr_pos_jacc, corr_pos_jacc_heur, pos_overlap, corr_pos_overlap, corr_iou = \
        compute_binary_stability_scores(bin_threshold, raw_predictions_collection)
    pearson_correlation, spearman_rank_correlation = compute_continuous_stability_scores(
        raw_predictions_collection)
    return pos_jacc, corr_pos_jacc, corr_pos_jacc_heur, pos_overlap, corr_pos_overlap, corr_iou, \
           pearson_correlation, spearman_rank_correlation
