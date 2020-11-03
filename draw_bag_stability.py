import pandas as pd
import seaborn as sns
sns.set()

import matplotlib.pyplot as plt

path = "C:/Users/mrtnr/Documents/results/"

bag_auc = [0.88, 0.85, 0.99]
apj = [0.42, 0.29, 0.56]
sp = [0.36, 0.44, 0.82]
data_names = ['X-Ray', 'MURA', 'Pascal VOC']
scores = ['apj', 's']

d = {'Dataset':['X-Ray', 'MURA',' Pascal VOC','X-Ray', 'MURA',' Pascal VOC'],
     'Stability Index':['APJ', 'APJ', 'APJ', 'S', 'S', 'S'],
     'Stability Score':[0.42, 0.29, 0.56,0.36, 0.44, 0.82],
     'Bag AUC': [0.88, 0.85, 0.99, 0.88, 0.85, 0.99]}
df = pd.DataFrame(data=d)

sns.set_style("whitegrid")
fig = plt.figure()
ax = sns.scatterplot(x='Stability Score', y='Bag AUC', data=df, style='Stability Index', hue='Dataset', s=100)
plt.xlim([0, 1])
plt.ylim([0, 1.1])
fig.savefig(path + 'scatter_.jpg', bbox_inches='tight')
plt.show()

