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


def plot_grouped_bar_population(df, file_name, res_path, findings_list):
    clas_0_labels = []
    clas_1_labels = []

    for clas in findings_list:
        clas_total = df[clas].value_counts()
        if clas_total.index[0] == np.float64(0):
            print("index is 0")
            print(clas_total.index[0])
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
    clas_1_labels  = np.asarray(clas_1_labels)


    x = np.arange(len(findings_list))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width/2, clas_0_labels, width, label='False')
    ax.bar(x + width/2, clas_1_labels, width, label='True')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Total observations')
    ax.set_title('Observations by diagnose and label')
    ax.set_xticks(x)
    ax.set_xticklabels(findings_list, fontsize=7)
    ax.legend()
    fig.savefig(res_path + 'population_data'+ '_' + file_name + '.jpg')
    # plt.show()