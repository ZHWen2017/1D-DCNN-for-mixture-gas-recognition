import numpy as np
# from MLPwithBN import *
import pickle
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools
import os
from mpl_toolkits.mplot3d import Axes3D

Max_steps = 2000

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# gas_path = 'C:/Users/Administrator/Desktop/resultPlot/dataPlot/gas_100.txt'
gas_path = '/home/wzh/PycharmProjects/data/multilabel/dataPlot/gas_100.txt'

def ConcatData(gasdata):
    aX0, aX1, aX2 = np.shape(gasdata.transpose(0, 2, 1))
    gasDataRe = []
    gasdata = gasdata.transpose(0, 2, 1)
    for i in range(aX0):
        gasMartix = gasdata[i]
        gas_temp = gasMartix[:, 0][::10]
        for j in range(aX2 - 1):
            gas_temp = np.concatenate((gas_temp, gasMartix[:, j+1][::10]))
        gasDataRe.append(gas_temp)
    return gasDataRe

def reSetLabel(gas_train):
    n0, m0 = np.shape(gas_train['label'])
    gas_train['label_co'] = []
    emty_index=[]
    for i in range(n0):
        if (gas_train['label'][i][0] == 0 and gas_train['label'][i][1] == 0
                and gas_train['label'][i][2] == 0):
            gas_train['label_co'].append(0)
        if (gas_train['label'][i][0] == 1 and gas_train['label'][i][1] == 0
                and gas_train['label'][i][2] == 0):
            gas_train['label_co'].append(1)
            # print('append 1: ', i)
        elif (gas_train['label'][i][0] == 0 and gas_train['label'][i][1] == 1
                and gas_train['label'][i][2] == 0):
            gas_train['label_co'].append(2)
            # print('append 2: ', i)
        elif (gas_train['label'][i][0] == 0 and gas_train['label'][i][1] == 0
                and gas_train['label'][i][2] == 1):
            gas_train['label_co'].append (3)
            # print('append 3: ', i)
        elif (gas_train['label'][i][0] == 1 and gas_train['label'][i][1] == 1
                and gas_train['label'][i][2] == 0):
            gas_train['label_co'].append(4)
            # print('append 4: ', i)
        elif (gas_train['label'][i][0] == 1 and gas_train['label'][i][1] == 0
                and gas_train['label'][i][2] == 1):
            gas_train['label_co'].append(5)
            # print('append 5: ', i)

    return gas_train['label_co']

f1 = open(gas_path, 'rb')
gas_data = pickle.load(f1)
f1.close()
print(gas_data['gas'].shape, np.shape(gas_data['label']))
for i in range(len(gas_data['gas'])):
    gas_temp = gas_data['gas'][i]
    mean_value = np.mean(gas_temp, axis=0)
    std_value = np.std(gas_temp, axis=0)
    # print(mean_value.shape, std_value.shape)
    for j in range(len(mean_value)):
        gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]
# print(gas_train['gas'][3].shape)



gas_data['label_'] = reSetLabel(gas_data)
gas_data['gas'] = np.reshape(np.array(gas_data['gas']), [593, -1])
pca = PCA(n_components=2)
pca.fit(gas_data['gas'])
gasData = np.array(pca.transform(gas_data['gas']))
gasLabel = np.array(gas_data['label_'])

print(gasData.shape, gasLabel.shape)

labelTups = [('Air', 0), ('Eth', 1), ('CO', 2), ('Met', 3), ('Eth-CO', 4), ('Eth-Met', 5)]
y = np.choose(gasLabel, [0, 1, 2, 3, 4, 5]).astype(np.float)

font = {'family': 'serif',
        'weight': 'normal',
        'size': 18,
        }

font1 = {'family': 'serif',
        'weight': 'normal',
        'size': 12,
        }

# plot first three PCA dimension
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
pca1 = PCA(n_components=3)
pca1.fit(gas_data['gas'])
X0 = np.array(pca1.transform(gas_data['gas']))
y0 = np.array(gas_data['label_'])

colors_label = [('y', 0, 'Air', 'o'),('b',1, 'Eth','*'),('m',2, 'CO', 'x'),
                ('c',3,'Met','p'),('r',4, 'Eth-CO','s'),('g',5, 'Eth-Met','d')]
features_3d = {}
for color, label, label_name,_ in colors_label:
    features_3d[label_name] = []
    for i,value in enumerate(y0):
        if value == label:
            features_3d[label_name].append(X0[i])
    features_3d[label_name] = np.array(features_3d[label_name])

for color, _, label_name, marker in colors_label:
    ax.scatter(features_3d[label_name][:, 0],
               features_3d[label_name][:, 1],
               features_3d[label_name][:, 2], c=color, marker='o',label=label_name, s=40)
ax.set_xlabel('X1', fontdict=font)
ax.set_ylabel('X2', fontdict=font)
ax.set_zlabel('X3', fontdict=font)
plt.legend(loc='center left', bbox_to_anchor=(0.93, 0.75), ncol=1, fontsize=14)
plt.savefig('./result_fig/3D_raw.svg')

fig1 = plt.figure(figsize=(10,8))
ax1 = fig1.add_subplot(111)
pca2 = PCA(n_components=3)
pca2.fit(gas_data['gas'])
X1 = np.array(pca2.transform(gas_data['gas']))
y1 = np.array(gas_data['label_'])

features_2d = {}
for color, label, label_name, _ in colors_label:
    features_2d[label_name] = []
    for i,value in enumerate(y1):
        if value == label:
            features_2d[label_name].append(X1[i])
    features_2d[label_name] = np.array(features_2d[label_name])
for color, _, label_name, _ in colors_label:
    ax1.scatter(features_2d[label_name][:, 0],
               features_2d[label_name][:, 1], c=color, marker='o', label=label_name)
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.8), ncol=1, fontsize=14)
ax1.set_xlabel('X1', fontdict=font)
ax1.set_ylabel('X2', fontdict=font)
ax1.spines['left'].set_linewidth(2.5)
ax1.spines['right'].set_linewidth(2.5)
ax1.spines['bottom'].set_linewidth(2.5)
ax1.spines['top'].set_linewidth(2.5)
plt.legend(loc='upper left',  fontsize=14)
# plt.savefig('./resultPlot/2D_feature.svg')



plt.show()



"""
# plot 2 dimension PCA
x_min, x_max = gasData[:, 0].min() - .5, gasData[:, 0].max() + .5
y_min, y_max = gasData[:, 1].min() - .5, gasData[:, 1].max() + .5
plt.figure(2, figsize=(14, 12))
plt.clf()
sc1 = plt.scatter(gasData[:, 0], gasData[:, 1], c=y, edgecolor='k')
plt.xlabel('X1', fontdict=font)
plt.ylabel('X2', fontdict=font)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
colors = [sc1.cmap(sc1.norm(i)) for i in [0, 1, 2, 3, 4, 5]]
custom_lines = [plt.Line2D([], [], ls="", marker='.',
                           mec='k', mfc=c, mew=.1, ms=20) for c in colors]
plt.legend(custom_lines, [lt[0] for lt in labelTups], loc='best', bbox_to_anchor=(1.0, 0.3), fontsize='large')
plt.savefig('2D_pca.eps')
plt.legend(loc='best', shadow=True, fontsize='medium', frameon=True, edgecolor='k')



# plot first three PCA dimension
pca1 = PCA(n_components=3)
pca1.fit(gas_data['gas'])
gasData1 = np.array(pca1.transform(gas_data['gas']))
gasLabel1 = np.array(gas_data['label_'])

fig = plt.figure(1, figsize=(12,10))
ax = Axes3D(fig, rect=[0, 0, .8, .8], elev=-150, azim=134)
sc = ax.scatter(gasData[:, 0], gasData1[:, 1], gasData1[:, 2], c=y,
           cmap='Spectral', edgecolor='k', s=40)


# ax.set_title("First three PCA directions", fontdict=font)
ax.set_xlabel("X1", fontdict=font)
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("X2", fontdict=font)
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("X3", fontdict=font)
ax.w_zaxis.set_ticklabels([])

colors = [sc.cmap(sc.norm(i)) for i in [0, 1, 2, 3, 4, 5]]
custom_lines = [plt.Line2D([], [], ls="", marker='.',
                           mec='k', mfc=c, mew=.1, ms=20) for c in colors]
ax.legend(custom_lines, [lt[0] for lt in labelTups], loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='large')
fig.savefig('3D_pca.eps')
plt.show()
"""
