import pandas as pd
from scipy.optimize import curve_fit

path = "C:/Users/s161590/Documents/Project_li/"

bag_auc = [0.74, 0.84, 0.99]
apj = [0.23, 0.66, 0.54]
sp = [0.32, 0.88, 0.82]
data_names = ['X-Ray', 'MURA', 'Pascal VOC']
scores = ['apj', 's']

d = {'Dataset':['X-Ray', 'MURA',' Pascal VOC','X-Ray', 'MURA',' Pascal VOC'],
     'Stability Index':['APJ', 'APJ', 'APJ', 'S', 'S', 'S'],
     'Stability Score':[0.23, 0.61, 0.54, 0.32, 0.84, 0.82],
     'Bag AUC': [0.72, 0.84, 0.99, 0.72, 0.84, 0.99]}
df = pd.DataFrame(data=d)

import matplotlib.cm as cm
import  numpy as np


def make_scatterplot(y_axis_collection, y_axis_title, x_axis_collection, x_axis_title, x_axis_collection2, res_path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    colors = cm.rainbow(np.linspace(0, 1, len(y_axis_collection)))
    for x, x2, y, color, sn in zip(x_axis_collection, x_axis_collection2, y_axis_collection, colors, data_names):
        # x, y = pearson_corr_col, spearman_corr_col
        ax.scatter(x, y, color=color, s=5, label=sn, marker='o')
        ax.scatter(x, y, color=color, edgecolors='none', s=70, marker='v',
                   vmin=0, vmax=1)
        ax.scatter(x2, y, color=color, edgecolors='none', s=70, marker='s',
                   vmin=0, vmax=1)
    ax.scatter(x, y, color=color, edgecolors='none', s=70, marker='v')
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Matplot scatter plot')
    plt.legend(loc=2)
    plt.show()
    fig.savefig(res_path +  'scatter_' + x_axis_title + '_' + y_axis_title + '.jpg', bbox_inches='tight')

    plt.close(fig)

make_scatterplot(bag_auc, 'Bag AUC', apj, 'score',sp, path)



import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

sns.set_style("whitegrid")
fig = plt.figure()
ax = sns.scatterplot(x='Stability Score', y='Bag AUC', data=df, style='Stability Index', hue='Dataset', s=100)
plt.xlim([0, 1])
plt.ylim([0, 1.1])
# ax = sns.scatterplot(x=apj, y=bag_auc, style=data_names)
fig.savefig(path + 'scatter_.jpg', bbox_inches='tight')
plt.show()


def exp_fit_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def make_scatterplot_with_errorbar(y_axis_collection, y_axis_title, x_axis_collection, x_axis_title, res_path, y_errors,
                                   fitting_curve = False, error_bar = False, bin_threshold_prefix =None, x_errors=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    # cmap2 = cm.get_cmap('tab20c')  # type: # matplotlib.colors.ListedColormap
    # colors = cmap2.colors  # type: # list
    # ax.set_prop_cycle(color=colors)

    # colors = cm.rainbow(np.linspace(0, 1, len(y_axis_collection)))
    if y_errors is None:
        y_errors = np.zeros(y_axis_collection.shape)
    if x_errors is None:
        x_errors = np.zeros(np.array(x_axis_collection).shape)
    for x, y, y_error_bar, x_error_bar in zip(x_axis_collection, y_axis_collection, y_errors, x_errors):
        # x, y = pearson_corr_col, spearman_corr_col
        ax.scatter(x, y, edgecolors='none', s=30, color="b")
        if (error_bar==True) and (y_errors is not None):
            ax.errorbar(x, y, xerr=x_error_bar, yerr=y_error_bar, color="b")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    if fitting_curve:
        popt, pcov = curve_fit(exp_fit_func, x_axis_collection, y_axis_collection, maxfev=1000)
        z = np.polyfit(x_axis_collection, y_axis_collection, 1)
        f = np.poly1d(z)

        z2 = np.polyfit(x_axis_collection, y_axis_collection, 2)
        f2 = np.poly1d(z2)
        # plt.plot(x_axis_collection, f(x), 'g-.', label="linear fit")
        # plt.plot(x_axis_collection, f2(x), 'b.', label="quadratic fit")
        print("I am the fitting curve: ")
        print(popt)
        plt.plot(np.sort(x_axis_collection), exp_fit_func(np.sort(x_axis_collection), *popt), 'r--',
                 label = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title('Matplot scatter plot')
    plt.legend(loc=(0.45, 0))
    plt.show()

    if bin_threshold_prefix is not None:
        fig.savefig(
            res_path + 'scatter_' + x_axis_title + '_' + y_axis_title + '_'+ str(bin_threshold_prefix)+'.jpg',
            bbox_inches='tight')
    else:
        fig.savefig(res_path +  'scatter_' + x_axis_title + '_' + y_axis_title + '.jpg', bbox_inches='tight')

    plt.close(fig)


mean_dice = [0.409994878, 0.414859883, 0.720072689, 0.618433118, 0.143922518, 0.724336751, 0.087368421, 0.419662813,
             0.367814213, 0.579860808, 0.596268861, 0, 0.186665339, 0.654044785, 0.420514651, 0.247710076, 0.751606661,
             0.43339558, 0.609695095, 0.475839622, 0.795331357, 0.140909091, 0.493320102, 0.71785501, 0.603801596,
             0.607286946, 0.776530369, 0.486685853]


std_dice = [0.294796378, 0.351121762, 0.087197918, 0.238660267, 0.271213835, 0.105318383, 0.107636404, 0.239172173,
            0.383663077,0.193531491, 0.25317776, 0, 0.229381448, 0.135023852,0.322713086 , 0.208616124  , 0.108702024,
            0.360300242 , 0.059184845, 0.053504954, 0.048793156, 0.281818182, 0.310137909, 0.114578536, 0.044253946,
            0.079263509, 0.03682584, 0.230254108]

spear = [0.196250233,0.20101299,0.510086515,0.327121696,0.156410751,0.505818172,0.163350094,0.14581734,0.062127372,
         0.314004224,0.529022833,0.107856274,0.04113739,0.567571585,0.206825162,0.118761008,0.53948546,0.30140256,
         0.575591121,0.327525236,0.546877971,0.152276854,0.298437046,0.439749032,0.681326301,0.614192396,0.589950351,
         0.401880582]

correctedjaccard = [0.11266614,0.098013985,0.389695436,0.229713858,0.001206246,0.437136034,-0.003931848,
                                0.115887753,0.097513695,0.252576314,0.222213004,0,0.053503688,0.404142458,0.11571405,
                                0.033392747,0.391272607,0.169253885,0.543924931,0.230939514,0.458249694,0,0.126549176,
                                0.374333695,0.427072986,0.398270644,0.488241154,0.169693694]


make_scatterplot_with_errorbar(mean_dice, 'mean DICE', correctedjaccard, 'DICE_mean adjusted Positive0 Jaccard',
                                   path, fitting_curve=False, y_errors=std_dice,
                                   x_errors=None, error_bar=True, bin_threshold_prefix=0)


make_scatterplot_with_errorbar(mean_dice, 'mean DICE', spear, 'DICE_mean_Spearman',
                                   res_path=path, fitting_curve=False, y_errors=std_dice,
                                   x_errors=None, error_bar=True, bin_threshold_prefix=0)