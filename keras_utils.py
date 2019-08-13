from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

#normalize between [-1, 1]
import pandas as pd

from custom_loss import test_compute_ground_truth_per_class_numpy
# from load_data import process_loaded_labels_tf


def normalize(im):
    return 2*(im/255) -1


def process_loaded_labels(label_col):
    newstr = (label_col.replace("[", "")).replace("]", "")
    return np.fromstring(newstr, dtype=np.ones((16, 16)).dtype, sep=' ').reshape(16, 16)


def process_loaded_labels_all_classes(label_col):
    res = []
    for ind in range(1, 15):
        np_array = (label_col.iloc[:, ind].values)
        print("in the function")
        print(np_array[0])
        print((np_array[0]))
        ret = process_loaded_labels(np_array[0])
        #
        res.append(res)
    return np.asarray(res)
        # newstr = (label_col.replace("[", "")).replace("]", "")
    # return np.fromstring(newstr, dtype=np.ones((16, 16)).dtype, sep=' ').reshape(16, 16)


def plot_train_validation(train_curve, val_curve, train_label, val_label,
                          title, y_axis, out_dir):
    plt.ioff()
    plt.clf()
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(train_curve)
    plt.plot(val_curve)

    plt.title(title)
    plt.ylabel(y_axis)
    plt.xlabel('epoch')
    # if third_curve is not None:
    #     plt.plot(third_curve)
    #     plt.legend([train_label, val_label, third_label], loc='upper left')
    # else:
    plt.legend([train_label, val_label ], loc='upper left')

    plt.savefig(out_dir + '/' + title + '.png')
    plt.clf()


def plot_ROC_curve(nr_class, fpr, tpr, roc_auc, out_dir):
    plt.figure()
    lw = 2
    plt.plot(fpr[nr_class], tpr[nr_class], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[nr_class])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(out_dir + '/roc_curve' + nr_class + '.png')
    plt.clf()


def prepare_data_visualization(df, findings_list):
    clas_0_labels = []
    clas_1_labels = []

    for clas in findings_list:
        clas_total = df[clas].value_counts()
        if clas_total.index[0] == np.float64(0):
            clas_0_labels.append(clas_total[0])
            if clas_total.shape[0] == 2:
                clas_1_labels.append(clas_total[1])
            else:
                clas_1_labels.append(0)
        else:
            clas_1_labels.append(clas_total[0])
            if clas_total.index[1] is not None:
                clas_0_labels.append(clas_total[1])
            else:
                clas_0_labels.append(0)
    clas_0_labels = np.asarray(clas_0_labels)
    clas_1_labels = np.asarray(clas_1_labels)
    return clas_0_labels, clas_1_labels


def plot_grouped_bar_population(df, file_name, res_path, findings_list):
    neg_labels, pos_labels = prepare_data_visualization(df, findings_list)

    x = np.arange(len(findings_list))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - width/2, neg_labels, width, label='False')
    ax.bar(x + width/2, pos_labels, width, label='True')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Total observations')
    ax.set_title('Observations by diagnose and label')
    ax.set_xticks(x)
    ax.set_xticklabels(findings_list, fontsize=8)
    ax.legend()
    fig.savefig(res_path + '/images/'+ 'population_data'+ '_' + file_name + '.jpg', bbox_inches='tight')
    # plt.show()
    plt.clf()


def plot_pie_population(df, file_name, res_path, findings_list):
    neg_labels, pos_labels = prepare_data_visualization(df, findings_list)

    x = np.arange(len(findings_list))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(2, 7, figsize=(18, 5))
    for i in range(0, 2):
        for j in range(0, 7):
           ax[i, j].pie([neg_labels[i*7+j], pos_labels[i*7+j]],autopct='%1.1f%%',
            shadow=True, startangle=90)
           ax[i, j].set_title(findings_list[i*7+j], {'fontsize': 9})

    fig.savefig(res_path + '/images/' + 'pie_chart' + '_' + file_name + '.jpg', bbox_inches='tight')
    plt.clf()


def visualize_population(df, file_name, res_path, findings_list):
    plot_grouped_bar_population(df, file_name, res_path, findings_list)
    plot_pie_population(df, file_name, res_path, findings_list)


def visualize_single_image_all_classes(xy_df_row, img_ind, results_path, prediction, img_prob,
                                       img_label, acc_per_class_sc,iou_score):
    for row in xy_df_row.values:
        labels_df = []

        #for each class
        for i in range(1, row.shape[0]):  # (15)
            g = process_loaded_labels(row[i])

            sum_active_patches, class_label_ground, has_bbox = test_compute_ground_truth_per_class_numpy(g, 16 * 16)
            # print("sum active patches: " + str(sum_active_patches))
            # print("class label: " + str(class_label_ground))
            # print("Has bbox:" + str(has_bbox))

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            ## show prediction active patches
            ax1 = plt.subplot(2, 2, 1)
            ax1.set_title('Original image', {'fontsize': 8})

            # img_dir = str(xy_df_row['Dir Path'][1])
            img_dir = Path(xy_df_row['Dir Path'].values[0]).__str__()
            img = plt.imread(img_dir)
            ax1.imshow(img, 'bone')

            ## PREDICTION
            ax2 = plt.subplot(2, 2, 2)
            ax2.set_title('Predictions: ' + xy_df_row.columns.values[i], {'fontsize': 8})
            im2 = ax2.imshow(prediction[0, :, :, i - 1], 'BuPu')
            fig.colorbar(im2, ax=ax2, norm=0)
            ax2.set_xlabel("Image prediction : " + str(img_prob[0, i - 1]))

            ## LABELS
            ax3 = plt.subplot(2, 2, 3)
            ax3.set_title('Labels: ' + xy_df_row.columns.values[i], {'fontsize': 8})
            ax3.set_xlabel(
                "Image label: " + str(class_label_ground) + str(img_label[0, i - 1]) + " Bbox available: " + str(
                    has_bbox))
            im3 = ax3.imshow(g)
            fig.colorbar(im3, ax=ax3, norm=0)

            ## BBOX of prediction and label
            ax4 = plt.subplot(2, 2, 4)
            ax4.set_title('Bounding boxes', {'fontsize': 8})

            y = (np.where(g == g.max()))[0]
            x = (np.where(g == g.max()))[1]

            upper_left_x = np.min(x)
            width = np.amax(x) - upper_left_x + 1
            upper_left_y = np.amin(y)
            height = np.amax(y) - upper_left_y + 1
            # todo: to draw using pyplot
            img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
                                        ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
            img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
                                        ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
            ax4.imshow(img, 'bone')
            # ax4.imshow(img4_labels, 'GnBu')
            pred_resized = np.kron(prediction[0, :, :, i - 1], np.ones((64, 64), dtype=float))
            img4_mask = ax4.imshow(pred_resized, 'BuPu', zorder=0, alpha=0.4)

            fig.text(0, 0, " Image prediction : " + str(img_prob[0, i - 1]) + '\n image label: ' +
                     str(img_label[0, i - 1]) + '\n IoU: ' + str(iou_score) +
                     '\n accuracy: ' + str(acc_per_class_sc), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)

            plt.tight_layout()
            fig.savefig(results_path + '/images/' + img_ind + '_' + xy_df_row.columns.values[i] + '.jpg',
                        bbox_inches='tight')
            plt.close(fig)


def save_evaluation_results(col_names, col_values,  file_name, out_dir,add_col=None, add_value=None):
    eval_df = pd.DataFrame()
    for i in range(0, len(col_names)):
        eval_df[col_names[i]] = pd.Series(col_values[i])
    if add_col is not None:
        eval_df[add_col] = add_value
    eval_df.to_csv(out_dir + '/' + file_name)


def save_accuracy_results(col_names, col_values, file_name, out_dir):
    localization_ind = [0, 1, 4, 8, 9, 10, 12, 13]
    eval_df = pd.DataFrame()
    for i in range(0, len(localization_ind)):
        eval_df[col_names[localization_ind[i]]] = pd.Series(col_values[i])
    #     for auc in range(len(auc_

    eval_df['Avg Accuracy'] = pd.Series(col_values[len(localization_ind)])
    eval_df.to_csv(out_dir + '/' + file_name)


def plot_grouped_bar_accuracy(acc_train, acc_val, acc_test, file_name, res_path, finding_list):
    localization_ind = [0, 1, 4, 8, 9, 10, 12, 13]

    finding_list_localization = [finding_list[i] for i in localization_ind]
    acc_tr_filter = [acc_train[i] for i in localization_ind]
    acc_val_filter = [acc_val[i] for i in localization_ind]
    acc_test_filter = [acc_test[i] for i in localization_ind]

    # neg_labels, pos_labels = prepare_data_visualization(df, finding_list)
    x = np.arange(len(finding_list_localization))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - width/3, acc_tr_filter, width, label='Training set')
    ax.bar(x - width, acc_val_filter, width, label='Validation set')
    ax.bar(x + width/3, acc_test_filter, width, label='Test set')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by diagnose and data set')
    ax.set_xticks(x)
    ax.set_xticklabels(finding_list_localization, fontsize=8)
    ax.legend()
    fig.savefig(res_path + '/images/'+ 'accuracy_'+ file_name + '.jpg', bbox_inches='tight')
    # plt.show()
    plt.clf()


def plot_grouped_bar_auc(auc_train, auc_val, auc_test, file_name, res_path, finding_list):
    # localization_ind = [0, 1, 4, 8, 9, 10, 12, 13]

    # finding_list_localization = [finding_list[i] for i in localization_ind]
        # acc_tr_filter = [auc_train[i] for i in localization_ind]
        # acc_val_filter = [auc_val[i] for i in localization_ind]
        # acc_test_filter = [auc_test[i] for i in localization_ind]

        # neg_labels, pos_labels = prepare_data_visualization(df, finding_list)
    x = np.arange(len(finding_list))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - width / 3, auc_train, width, label='Training set')
    ax.bar(x - width, auc_val, width, label='Validation set')
    ax.bar(x + width / 3, auc_test, width, label='Test set')

      # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUC')
    ax.set_title('AUC by diagnose and data set')
    ax.set_xticks(x)
    ax.set_xticklabels(finding_list, fontsize=8)
    ax.legend()
    fig.savefig(res_path + '/images/' + 'auc_' + file_name + '.jpg', bbox_inches='tight')
        # plt.show()
    plt.clf()