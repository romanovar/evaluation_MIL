import matplotlib.pyplot as plt


#normalize between [-1, 1]
def normalize(im):
    return 2*(im/255) -1


def plot_train_validation(train_curve, val_curve, title, y_axis, out_dir):
    plt.ioff()
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(train_curve)
    plt.plot(val_curve)
    plt.title(title)
    plt.ylabel(y_axis)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig(out_dir + '/' + title + '.png')
    plt.clf()