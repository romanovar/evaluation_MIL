import matplotlib.pyplot as plt
import numpy as np

#normalize between [-1, 1]
def normalize(im):
    return 2*(im/255) -1


def plot_train_validation(train_curve, val_curve, third_curve, train_label, val_label, third_label,
                          title, y_axis, out_dir):
    plt.ioff()
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(train_curve)
    plt.plot(val_curve)

    plt.title(title)
    plt.ylabel(y_axis)
    plt.xlabel('epoch')
    if third_curve is not None:
        plt.plot(third_curve)
        plt.legend([train_label, val_label, third_label], loc='upper left')
    else:
        plt.legend([train_label, val_label, ], loc='upper left')

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


def plot_pie_population(df, file_name, res_path, findings_list):
    neg_labels, pos_labels = prepare_data_visualization(df, findings_list)

    x = np.arange(len(findings_list))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(2, 7, figsize=(15, 8))
    for i in range(0, 2):
        for j in range(0, 7):
           ax[i, j].pie([neg_labels[i*7+j], pos_labels[i*7+j]],autopct='%1.1f%%',
            shadow=True, startangle=90)
           ax[i, j].set_title(findings_list[i*7+j], {'fontsize': 9})
           # ax[i,j].legend()
    # fig.legend()

# fig, axes = plt.subplots(1, 2)
    # axes[1].pie(neg_labels, labels=findings_list, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    # axes[1].axis('equal')
    #
    # axes[0, 1].pie(pos_labels, labels=findings_list, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    # axes[0, 1].axis('equal')


    fig.savefig(res_path + '/images/' + 'pie_chart' + '_' + file_name + '.jpg', bbox_inches='tight')


def visualize_population(df, file_name, res_path, findings_list):
    plot_grouped_bar_population(df, file_name, res_path, findings_list)
    plot_pie_population(df, file_name, res_path, findings_list)
