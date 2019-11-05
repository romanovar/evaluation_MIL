import PIL
from pathlib import Path
import matplotlib
from PIL import ImageDraw
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.utils import resample
import tensorflow as tf
#normalize between [-1, 1]
import pandas as pd
import matplotlib.cm as cm
from cnn.nn_architecture.custom_loss import test_compute_ground_truth_per_class_numpy, compute_ground_truth
import matplotlib
import seaborn as sns

from stability.preprocessor.preprocessing import binarize_predictions
from stability.stability_2classifiers.scores_2classifiers import positive_Jaccard_index_batch, \
    corrected_Jaccard_pigeonhole, corrected_positive_Jaccard, overlap_coefficient, corrected_overlap_coefficient, \
    corrected_IOU


def get_image_index_from_pathstring(string_path):
    '''
    :param string_path: string of the directory path where the image is
    :return: returns the image index
    0000PAT_IND.png = 4 symbols + 3 symbols + _ + 8 first number = 4+4+8 = 16 symbols and skipping '.png'
    '''

    return string_path[-16:-4]


def visualize_single_image_1class_2predictions(img_ind_coll,labels_coll,  raw_predictions_coll, classifier_name,
                                               auc_score, raw_predictions_coll2, classifier_name2, auc_score2,
                                               img_path, results_path, class_name,
                                               image_title_suffix,  jaccard_ind, corrected_jaccard,
                                               corrected_jaccard_pigeonhole,  corrected_iou, overlap_ind, corr_overlap,
                                               pearson_corr_coef, spearman_corr_coef):
    # for each row/observation in the batch
    for ind in range(0, img_ind_coll.shape[0]):
        print(ind)
        threshold_transparency = 0.01

        instance_label_gt = labels_coll[ind, :, :,0]
        img_ind = img_ind_coll[ind]
        raw_prediction = raw_predictions_coll[ind, :, :, 0]
        auc = auc_score[ind]

        raw_prediction2 = raw_predictions_coll2[ind, :, :, 0]
        auc2 = auc_score2[ind]

        img_dir = Path(img_path + get_image_index_from_pathstring(img_ind) + '.png').__str__()
        img = plt.imread(img_dir)

        scale_width = int(img.shape[1]/16)
        scale_height =int(img.shape[0]/16)
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))


        ## PREDICTIONS: BBOX of prediction and label
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('Prediction Classifier '+ classifier_name, {'fontsize': 9})
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

        pred_resized = np.kron(raw_prediction, np.ones((64, 64), dtype=float))
        pred_resized[pred_resized < threshold_transparency] = np.nan
        img1_mask = ax1.imshow(pred_resized, 'BuPu', zorder=0, alpha = 0.8, vmin=0, vmax=1)
        ax1.set_xlabel("AUC instance score: "+ ("{0:.3f}".format(auc)))
        fig.colorbar(img1_mask,ax=ax1, fraction=0.046)

        fig.text(-0.2, 0.5, '\n Only patches with prediction score above '+ str(threshold_transparency) +" are shown! ",
                 horizontalalignment='center',
                 verticalalignment='center', fontsize=9)

        ## PREDICTIONS: BBOX of prediction and label
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('Prediction Classifier '+ classifier_name2, {'fontsize': 9})

        ax2.imshow(img_bbox, 'bone')
        pred_resized2 = np.kron(raw_prediction2, np.ones((64, 64), dtype=float))
        pred_resized2[pred_resized2 < threshold_transparency] = np.nan
        img2_mask = ax2.imshow(pred_resized2, 'BuPu', zorder=0, alpha=0.8,  vmin=0, vmax=1)
        ax2.set_xlabel("AUC instance score: " + ("{0:.3f}".format(auc2)))
        fig.colorbar(img2_mask, ax=ax2, fraction=0.046)

        # ## LABELS
        # ax3 = plt.subplot(2, 2, 3)
        # ax3.set_title('Labels: ' + class_name, {'fontsize': 8})
        # im3 = ax3.imshow(instance_label_gt, vmin=0, vmax=1)
        # fig.colorbar(im3, ax=ax3)
        #
        # ## BBOX of prediction and label
        # ax4 = plt.subplot(2, 2, 4)
        # ax4.set_title('Predictions classifer '+ classifier_name2, {'fontsize': 8})
        #
        # y = (np.where(instance_label_gt == instance_label_gt.max()))[0]
        # x = (np.where(instance_label_gt == instance_label_gt.max()))[1]
        #
        # upper_left_x = np.min(x)
        # # width = np.amax(x) - upper_left_x + 1
        # upper_left_y = np.amin(y)
        # # height = np.amax(y) - upper_left_y + 1
        # # todo: to draw using pyplot
        # img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
        #                             ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
        # img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
        #                             ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
        # ax4.imshow(img, 'bone')
        # pred_resized = np.kron(raw_prediction2, np.ones((64, 64), dtype=float))
        # img4_mask = ax4.imshow(pred_resized, 'BuPu', zorder=0, alpha=0.4)

        fig.text(-0.2, 0.43, '\n Overlap index: ' + "{:.2f}".format(overlap_ind[ind])+
                 '\n Corrected overlap index: ' + "{:.2f}".format(corr_overlap[ind])+
                  '\n positive Jaccard distance: ' + "{:.2f}".format(jaccard_ind[ind])+
                 '\n Corrected positive Jaccard distance: ' + "{:.2f}".format(corrected_jaccard[ind])+
                 '\n Corrected positive Jaccard using pigeonhole: ' + "{:.2f}".format(corrected_jaccard_pigeonhole[ind])+
                 '\n Corrected IoU: ' + "{:.2f}".format(corrected_iou[ind]) +
                 '\n Pearson correlation coefficient: ' + "{:.2f}".format(pearson_corr_coef[ind])+
                 '\n Spearman rank correlation coefficient: ' + "{:.2f}".format(spearman_corr_coef[ind]),
                 horizontalalignment='center', verticalalignment='center', fontsize=9)

        plt.tight_layout()
        fig.savefig(results_path + get_image_index_from_pathstring(img_ind) + '_' + class_name + image_title_suffix + '.jpg',
                    bbox_inches='tight')
        plt.close(fig)


def visualize_single_image_1class_5classifiers(img_ind_coll, labels_coll, raw_predictions_coll, img_path,
                                               results_path,
                                               class_name,
                                               image_title_suffix):

    for ind in range(0, img_ind_coll[0].shape[0]):
        print(ind)
        threshold_transparency = 0.01

        instance_label_gt = labels_coll[0][ind, :, :, 0]
        img_ind = img_ind_coll[0][ind]
        raw_prediction = raw_predictions_coll[0][ind, :, :, 0]
        # auc = auc_score[ind]

        raw_prediction2 = raw_predictions_coll[1][ind, :, :, 0]
        # auc2 = auc_score2[ind]

        img_dir = Path(img_path + get_image_index_from_pathstring(img_ind) + '.png').__str__()
        img = plt.imread(img_dir)

        scale_width = int(img.shape[1] / 16)
        scale_height = int(img.shape[0] / 16)
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
        img1_mask = ax1.imshow(pred_resized, 'BuPu', zorder=0, alpha=0.8, vmin=0, vmax=1)
        # ax1.set_xlabel("AUC instance score: "+ ("{0:.3f}".format(auc)))
        fig.colorbar(img1_mask, ax=ax1, fraction=0.046)

        fig.text(-0.2, 0.5,
                 '\n Only patches with prediction score above ' + str(threshold_transparency) + " are shown! ",
                 horizontalalignment='center',
                 verticalalignment='center', fontsize=9)

        ## SUB-GRAPH 2
        ax2 = plt.subplot(2, 3, 2)
        ax2.set_title('Predictions Classifier 2', {'fontsize': 9})

        ax2.imshow(img_bbox, 'bone')
        pred_resized2 = np.kron(raw_predictions_coll[1][ind, :, :, 0], np.ones((64, 64), dtype=float))
        pred_resized2[pred_resized2 < threshold_transparency] = np.nan
        img2_mask = ax2.imshow(pred_resized2, 'BuPu', zorder=0, alpha=0.8, vmin=0, vmax=1)
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
        pred_resized4 = np.kron(raw_predictions_coll[3][ind, :, :, 0], np.ones((64, 64), dtype=float))
        pred_resized4[pred_resized4 < threshold_transparency] = np.nan
        img4_mask = ax4.imshow(pred_resized4, 'BuPu', zorder=0, alpha=0.8, vmin=0, vmax=1)
        # ax2.set_xlabel("AUC instance score: " + ("{0:.3f}".format(auc2)))
        fig.colorbar(img4_mask, ax=ax4, fraction=0.046)

        ## SUB-GRAPH 5
        ax5 = plt.subplot(2, 3, 5)
        ax5.set_title('Predictions Classifier 5', {'fontsize': 9})

        ax5.imshow(img_bbox, 'bone')
        pred_resized5 = np.kron(raw_predictions_coll[4][ind, :, :, 0], np.ones((64, 64), dtype=float))
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
        fig.savefig(results_path + get_image_index_from_pathstring(
            img_ind) + '_' + class_name + image_title_suffix + '.jpg',
                    bbox_inches='tight')
        plt.close(fig)



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


def plot_line_graph(line1, label1, line2, label2, line3, label3, line4, label4, line5, label5,  line6, label6,
                    x_axis_data, x_label, results_path, fig_name, text_string):

    fig = plt.figure()
    fig.text(0, 0, text_string, horizontalalignment='center', verticalalignment='center', fontsize=9)
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    plt.plot(x_axis_data, line1, 'g', label = label1)
    plt.plot(x_axis_data, line2, 'b', label=label2)
    if line5 is not None:
        plt.plot(x_axis_data, line5, '-r', label=label5)
        if line3 is not None:
            plt.plot(x_axis_data, line3, ':g', label=label3)
            if line4 is not None:
                plt.plot(x_axis_data, line4, ':b', label=label4)
                if line6 is not None:
                    plt.plot(x_axis_data, line6, 'r--', label=label6)
    # plt.tick_params(labelsize=2)
    plt.rc('legend', fontsize=7)
    plt.ylabel('score')
    plt.xlabel(x_label)
    plt.legend()
    plt.show()
    fig.savefig(
        results_path +  fig_name + '.jpg',
        bbox_inches='tight')
    plt.close(fig)
    plt.clf()