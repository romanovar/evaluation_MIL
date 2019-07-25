import matplotlib.pyplot as plt


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