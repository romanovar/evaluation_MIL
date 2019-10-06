import PIL
from pathlib import Path
import matplotlib
from PIL import ImageDraw
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.utils import resample
import tensorflow as tf
#normalize between [-1, 1]
import pandas as pd
import matplotlib.cm as cm
from custom_loss import test_compute_ground_truth_per_class_numpy, compute_ground_truth
import matplotlib

def normalize(im):
    return 2*(im/255) -1


def process_loaded_labels(label_col):
    newstr = (label_col.replace("[", "")).replace("]", "")
    return np.fromstring(newstr, dtype=np.ones((16, 16)).dtype, sep=' ').reshape(16, 16)


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
    xint = range(0, len(train_curve))

    plt.xticks(xint)
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
    fig.savefig(res_path + 'population_data'+ '_' + file_name + '.jpg', bbox_inches='tight')
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

    fig.savefig(res_path +  'pie_chart' + '_' + file_name + '.jpg', bbox_inches='tight')
    plt.clf()


def visualize_population(df, file_name, res_path, findings_list):
    plot_grouped_bar_population(df, file_name, res_path, findings_list)
    plot_pie_population(df, file_name, res_path, findings_list)


def get_image_index_from_pathstring(string_path):
    '''
    :param string_path: string of the directory path where the image is
    :return: returns the image index
    0000PAT_IND.png = 4 symbols + 3 symbols + _ + 8 first number = 4+4+8 = 16 symbols and skipping '.png'
    '''

    return string_path[-16:-4]


# The function is redundant and can be more efficient
# it is uses same code as in the generator - so it is a test that everything works adequately
def prepare_labels_all_classes(instance, processed_y):
    instances_classes =[]
    for i in range(1, instance.shape[0]):  # (15)
        if processed_y:
            g = process_loaded_labels(instance[i])
            instances_classes.append(g)
        else:
            instances_classes.append(instance[i])
    return np.transpose(np.asarray(instances_classes), [1, 2, 0])


def visualize_single_image_all_classes(batch_df, img_ind, results_path, batch_predictions, batch_img_prob,
                                       img_label, skip_process):
    ind = 0
    # for each row/observation in the batch
    for row in batch_df.values:
        labels_df = []

        instance_labels_gt = prepare_labels_all_classes(row, skip_process)
        sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_labels_gt, 256,1)

        #for each class
        for i in range(1, row.shape[0]):  # (15)

            init_op = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init_op)

                # slicing on index 0 it will have shape of (1, class_number)
                class_gt = class_label_ground_truth[0, i - 1].eval()
                # instance labels have shape of [16, 16, class_nr]
                instance_gt = instance_labels_gt[:, :, i-1]
                total_active_patches = sum_active_patches[0, i-1].eval()
                has_segmentation = has_bbox[0, i-1].eval()
                img_probab = batch_img_prob[ind].eval()
                img_label_test = img_label[ind].eval()

                assert img_label_test == class_gt

                print(img_probab)
                print(batch_predictions[ind, :, :, i-1])
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            ## show prediction active patches
            ax1 = plt.subplot(2, 2, 1)
            ax1.set_title('Original image', {'fontsize': 8})

            img_dir = Path(batch_df['Dir Path'].values[0]).__str__()
            img = plt.imread(img_dir)
            ax1.imshow(img, 'bone')

            ## PREDICTION
            ax2 = plt.subplot(2, 2, 2)
            ax2.set_title('Predictions: ' + batch_df.columns.values[i], {'fontsize': 8})
            im2 = ax2.imshow(batch_predictions[ind, :, :, i - 1], 'BuPu', vmin=0, vmax=1)
            fig.colorbar(im2,ax=ax2)
            ax2.set_xlabel("Image prediction : " + str(img_probab))

            ## LABELS
            ax3 = plt.subplot(2, 2, 3)
            # class_gt = class_label_ground_truth.eval()
            ax3.set_title('Labels: ' + batch_df.columns.values[i], {'fontsize': 8})
            ax3.set_xlabel(
                "Image label: " + str(class_gt) + str(img_label_test) + " Bbox available: " + str(
                    has_segmentation))
            im3 = ax3.imshow(instance_gt, vmin=0, vmax=1)
            fig.colorbar(im3, ax=ax3)

            ## BBOX of prediction and label
            ax4 = plt.subplot(2, 2, 4)
            ax4.set_title('Bounding boxes', {'fontsize': 8})

            y = (np.where(instance_gt == instance_gt.max()))[0]
            x = (np.where(instance_gt == instance_gt.max()))[1]

            upper_left_x = np.min(x)
            # width = np.amax(x) - upper_left_x + 1
            upper_left_y = np.amin(y)
            # height = np.amax(y) - upper_left_y + 1
            # todo: to draw using pyplot
            img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
                                        ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
            img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
                                        ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
            ax4.imshow(img, 'bone')
            # ax4.imshow(img4_labels, 'GnBu')
            pred_resized = np.kron(batch_predictions[ind, :, :, i - 1], np.ones((64, 64), dtype=float))
            img4_mask = ax4.imshow(pred_resized, 'BuPu', zorder=0, alpha=0.4)

            fig.text(0, 0, " Image prediction : " + str(img_probab) + '\n image label: ' +
                     str(class_gt), horizontalalignment='center',
                     verticalalignment='center', fontsize=9)

            plt.tight_layout()
            fig.savefig(results_path + get_image_index_from_pathstring(img_ind[ind]) + '_' + batch_df.columns.values[i][:-4] + '.jpg',
                        bbox_inches='tight')
            plt.close(fig)
        ind += 1


def visualize_single_image_1class(img_ind_coll, raw_predictions_coll, labels_coll, img_path, results_path, class_name,
                                  image_title_suffix, auc_score, jaccard_ind, corr_coef ):
    # for each row/observation in the batch
    for ind in range(0, img_ind_coll.shape[0]):
        labels_df = []

        instance_label_gt = labels_coll[ind, :, :,0]
        raw_prediction = raw_predictions_coll[ind, :, :, 0]
        img_ind = img_ind_coll[ind]
        print("im here: "+get_image_index_from_pathstring(img_ind))
        # sum_active_patches, class_label_ground_truth, has_bbox = compute_ground_truth(instance_label_gt, 256,1)
        #
        #         #for each class
        #         # for i in range(1, row.shape[0]):  # (15)
        #
        #         # init_op = tf.global_variables_initializer()
        #         # with tf.Session() as sess:
        #         #     sess.run(init_op)
        #         #
        #         #     # slicing on index 0 it will have shape of (1, class_number)
        #         #     class_gt = class_label_ground_truth[0, i - 1].eval()
        #         #     # instance labels have shape of [16, 16, class_nr]
        #         #     instance_gt = instance_label_gt[:, :, i-1]
        #         #     total_active_patches = sum_active_patches[0, i-1].eval()
        #         #     has_segmentation = has_bbox[0, i-1].eval()
        #         #     img_probab = batch_img_prob[ind].eval()
        #         #     img_label_test = img_label[ind].eval()
        #         #
        #         #     assert img_label_test == class_gt
        #         #
        #         #     print(img_probab)
        #         #     print(batch_predictions[ind, :, :, i-1])

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        ## show prediction active patches
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title('Original image', {'fontsize': 8})
        #fix
        # img_dir = Path(batch_df['Dir Path'].values[0]).__str__()
        img_dir = Path(img_path+get_image_index_from_pathstring(img_ind)+'.png').__str__()
        img = plt.imread(img_dir)
        ax1.imshow(img, 'bone')

        ## PREDICTION
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title('Predictions: ' + class_name, {'fontsize': 8})
        im2 = ax2.imshow(raw_prediction, 'BuPu', vmin=0, vmax=1)
        fig.colorbar(im2,ax=ax2)
        # ax2.set_xlabel("Image prediction : " + str(img_probab))

        ## LABELS
        ax3 = plt.subplot(2, 2, 3)
        # class_gt = class_label_ground_truth.eval()
        ax3.set_title('Labels: ' + class_name, {'fontsize': 8})
        # ax3.set_xlabel(
        #     "Image label: " + str(class_gt) + str(img_label_test) + " Bbox available: " + str(
        #         has_segmentation))
        im3 = ax3.imshow(instance_label_gt, vmin=0, vmax=1)
        fig.colorbar(im3, ax=ax3)

        ## BBOX of prediction and label
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_title('Bounding boxes', {'fontsize': 8})

        y = (np.where(instance_label_gt == instance_label_gt.max()))[0]
        x = (np.where(instance_label_gt == instance_label_gt.max()))[1]

        upper_left_x = np.min(x)
        # width = np.amax(x) - upper_left_x + 1
        upper_left_y = np.amin(y)
        # height = np.amax(y) - upper_left_y + 1
        # todo: to draw using pyplot
        img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
                                    ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
        img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
                                    ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
        ax4.imshow(img, 'bone')
        # ax4.imshow(img4_labels, 'GnBu')
        pred_resized = np.kron(raw_prediction, np.ones((64, 64), dtype=float))
        img4_mask = ax4.imshow(pred_resized, 'BuPu', zorder=0, alpha=0.4)

        fig.text(0, 0, " Instance AUC: " + str(auc_score[ind]) + '\n Jaccard index: ' +
                 str(jaccard_ind[ind])+ '\n Correlation coefficient: ' +
                 str(corr_coef[ind]), horizontalalignment='center',
                 verticalalignment='center', fontsize=9)

        plt.tight_layout()
        fig.savefig(results_path + get_image_index_from_pathstring(img_ind) + '_' + class_name + image_title_suffix + '.jpg',
                    bbox_inches='tight')
        plt.close(fig)
        # ind += 1


def save_evaluation_results(col_names, col_values,  file_name, out_dir,add_col=None, add_value=None):
    eval_df = pd.DataFrame()
    for i in range(0, len(col_names)):
        eval_df[col_names[i]] = pd.Series(col_values[i])
    if add_col is not None:
        eval_df[add_col] = add_value
    eval_df.to_csv(out_dir + '/' + file_name)


# def save_accuracy_results(col_names, col_values, file_name, out_dir):
#     localization_ind = [0, 1, 4, 8, 9, 10, 12, 13]
#     eval_df = pd.DataFrame()
#     for i in range(0, len(localization_ind)):
#         eval_df[col_names[localization_ind[i]]] = pd.Series(col_values[i])
#     #     for auc in range(len(auc_
#
#     eval_df['Avg Accuracy'] = pd.Series(col_values[len(localization_ind)])
#     eval_df.to_csv(out_dir + '/' + file_name)


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


def visualize_single_image_1class_2predictions(img_ind_coll,labels_coll,  raw_predictions_coll, classifier_name,
                                               auc_score, raw_predictions_coll2, classifier_name2, auc_score2,
                                               img_path, results_path, class_name,
                                               image_title_suffix,  jaccard_ind, corr_coef):
    # for each row/observation in the batch
    for ind in range(0, img_ind_coll.shape[0]):
        threshold_transparency = 0.01

        instance_label_gt = labels_coll[ind, :, :,0]
        img_ind = img_ind_coll[ind]
        raw_prediction = raw_predictions_coll[ind, :, :, 0]
        auc = auc_score[ind]

        raw_prediction2 = raw_predictions_coll2[ind, :, :, 0]
        auc2 = auc_score2[ind]

        # fix
        # img_dir = Path(batch_df['Dir Path'].values[0]).__str__()
        img_dir = Path(img_path + get_image_index_from_pathstring(img_ind) + '.png').__str__()
        img = plt.imread(img_dir)


        # todo: introduce scale factor
        # height, width = img.shape[:2]
        scale_width = int(img.shape[1]/16)
        scale_height =int(img.shape[0]/16)
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        ## show prediction active patches



        ## PREDICTIONS: BBOX of prediction and label
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title('Prediction Classifier '+ classifier_name, {'fontsize': 9})
        y = (np.where(instance_label_gt == instance_label_gt.max()))[0]
        x = (np.where(instance_label_gt == instance_label_gt.max()))[1]
        bottom_x = np.amin(x)
        bottom_y = np.amax(y)
        upper_left_x = np.min(x)

        upper_left_y = np.amin(y)
        width = np.amax(x) - upper_left_x
        height = np.amax(y) - upper_left_y
        print(np.min(x))
        print(np.min(y))
        print(np.max(x))
        print(np.max(y))
        # https: // pillow.readthedocs.io / en / 4.2.x / reference / ImageDraw.html  # PIL.ImageDraw.PIL.ImageDraw.Draw.rectangle
        # ImageDraw.ImageDraw.rectangle([(5, 5), (6, 6)], fill=None, outline=(255, 0, 0))

        # matplotlib
        # bbox= matplotlib.patches.Rectangle((upper_left_x , upper_left_y ),
        #                             (np.amax(x) + 1) , (np.amax(y) + 1),linewidth=15,edgecolor='r')
        # ax1.add_patch(bbox)


        # OPENCV
        img_bbox = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_bbox, (upper_left_x * scale_width, upper_left_y * scale_height),
                                 ((np.amax(x) + 1) * scale_width, (np.amax(y) + 1) * scale_height), (125, 0, 0), 5)

        # img_b = img_b*(255.0)

        # ax4.imshow(img4_labels, 'GnBu')
        # ax1.imshow(img_rbg, 'bone')
        # tag_img = cv2.imread(img_b, 1)

        ax1.imshow(img_bbox)
        red_patch = matplotlib.patches.Patch(color='red', label='Ground truth annotation')
        # ax1.legend(handles=[red_patch]) bbox_to_anchor=(0, 1),
        plt.legend(handles=[red_patch], bbox_to_anchor=(-0.2, -0.2), loc='lower right', borderaxespad=0.)

        pred_resized = np.kron(raw_prediction, np.ones((64, 64), dtype=float))
        pred_resized[pred_resized < threshold_transparency] = np.nan
        img1_mask = ax1.imshow(pred_resized, 'BuPu', zorder=0, alpha = 0.8)
        ax1.set_xlabel("AUC instance score: "+ ("{0:.3f}".format(auc)))
        fig.colorbar(img1_mask,ax=ax1)

        fig.text(-0.2, 0.5, '\n Only patches with prediction score above '+ str(threshold_transparency) +" are shown! ",
                 horizontalalignment='center',
                 verticalalignment='center', fontsize=9)
        # ## PREDICTION
        # ax2 = plt.subplot(2, 2, 2)
        # ax2.set_title('Predictions: ' + class_name, {'fontsize': 8})
        # im2 = ax2.imshow(raw_prediction, 'BuPu', vmin=0, vmax=1)
        # fig.colorbar(im2,ax=ax2)
        #
        # # ax2.set_xlabel("Image prediction : " + str(img_probab))

        ## PREDICTIONS: BBOX of prediction and label
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title('Prediction Classifier '+ classifier_name2, {'fontsize': 9})

        ax2.imshow(img_bbox, 'bone')
        pred_resized2 = np.kron(raw_prediction2, np.ones((64, 64), dtype=float))
        pred_resized2[pred_resized2 < threshold_transparency] = np.nan
        img2_mask = ax2.imshow(pred_resized2, 'RdPu', zorder=0, alpha=0.8,  vmin=0, vmax=1)
        ax2.set_xlabel("AUC instance score: " + ("{0:.3f}".format(auc2)))
        fig.colorbar(img2_mask, ax=ax2)

        ## LABELS
        ax3 = plt.subplot(2, 2, 3)
        # class_gt = class_label_ground_truth.eval()
        ax3.set_title('Labels: ' + class_name, {'fontsize': 8})
        # ax3.set_xlabel(
        #     "Image label: " + str(class_gt) + str(img_label_test) + " Bbox available: " + str(
        #         has_segmentation))
        im3 = ax3.imshow(instance_label_gt, vmin=0, vmax=1)
        fig.colorbar(im3, ax=ax3)

        ## BBOX of prediction and label
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_title('Predictions classifer '+ classifier_name2, {'fontsize': 8})

        y = (np.where(instance_label_gt == instance_label_gt.max()))[0]
        x = (np.where(instance_label_gt == instance_label_gt.max()))[1]

        upper_left_x = np.min(x)
        # width = np.amax(x) - upper_left_x + 1
        upper_left_y = np.amin(y)
        # height = np.amax(y) - upper_left_y + 1
        # todo: to draw using pyplot
        img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
                                    ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
        img4_labels = cv2.rectangle(img, (upper_left_x * 64, upper_left_y * 64),
                                    ((np.amax(x) + 1) * 64, (np.amax(y) + 1) * 64), (0, 255, 0), 5)
        ax4.imshow(img, 'bone')
        # ax4.imshow(img4_labels, 'GnBu')
        pred_resized = np.kron(raw_prediction2, np.ones((64, 64), dtype=float))
        img4_mask = ax4.imshow(pred_resized, 'BuPu', zorder=0, alpha=0.4)

        fig.text(0, 0, '\n Jaccard index: ' +
                 str(jaccard_ind[ind])+ '\n Correlation coefficient: ' +
                 str(corr_coef[ind]), horizontalalignment='center',
                 verticalalignment='center', fontsize=9)

        plt.tight_layout()
        fig.savefig(results_path + get_image_index_from_pathstring(img_ind) + '_' + class_name + image_title_suffix + '.jpg',
                    bbox_inches='tight')
        plt.close(fig)


##################### BOOTSTRAP OVERLAP #########################################################


def return_rows_to_drop_bootstrap(init_train_df, overlap_pat_ratio, seed):
    obs_indices = init_train_df.index.values
    samples_to_drop = np.math.floor((1 - overlap_pat_ratio) * init_train_df.shape[0])
    return resample(obs_indices, n_samples=samples_to_drop, replace=False, random_state=seed)


def drop_rows(init_train_df, ind_to_drop):
    return init_train_df.drop(ind_to_drop)


def create_overlap_set_bootstrap(init_train_df, overlap_pat_ratio, seed):
    rows_drop = return_rows_to_drop_bootstrap(init_train_df, overlap_pat_ratio, seed)
    return drop_rows(init_train_df, rows_drop)


def save_evaluation_results_1class(col_names, col_values,  file_name, out_dir,add_col=None, add_value=None):
    eval_df = pd.DataFrame()
    for i in range(0, len(col_names)):
        eval_df[col_names[i]] = pd.Series(col_values[i])
    if add_col is not None:
        eval_df[add_col] = add_value
    eval_df.to_csv(out_dir + '/' + file_name)