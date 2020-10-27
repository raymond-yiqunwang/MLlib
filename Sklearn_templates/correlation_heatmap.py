import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import math
import os

#data = pd.read_csv("../data/data_all_float.csv", header=0, index_col=None, sep=';')

# drop instances with NaN
drop_list = []
for idx in data.index:
    if True in [math.isnan(x) for x in data.loc[idx].values]:
        drop_list.append(idx)
data = data.drop(drop_list)
print('The shape of data: ', data.shape)

corr = data.corr()

fig = plt.figure(num=None, figsize=(40, 40), dpi=80, facecolor='w', edgecolor='w')
colormap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
ylabel = list(corr.columns)
ylabel[0] = ''
ax = sns.heatmap(corr, cmap=colormap, mask=mask, annot=True, fmt=".2f", xticklabels=corr.columns[:-1], yticklabels=ylabel)
plt.title(label="Correlation Heatmap of All Features", loc='center', fontdict={'fontname':'DejaVu Sans', 'size':'24', 'color':'black', 'weight':'bold',
              'verticalalignment':'bottom'})
ax.tick_params(axis='x', rotation=60, labelsize=10)
ax.tick_params(axis='y', labelsize=13)
plt.show()
fig.savefig("heatmap_mask.png", dpi=200)
