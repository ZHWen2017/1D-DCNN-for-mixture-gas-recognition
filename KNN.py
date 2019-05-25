import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
f = open("/home/wzh/PycharmProjects/data/multilabel/dataPlot/gas_100.txt", "rb")
gas = pickle.load(f)
# gas_test['label'] = np.array(gas_test['label'])
print(np.shape(gas['label']), np.shape(gas['gas']))
f.close()
Gas = {}
Gas['label'] = gas['label']
Gas['data'] = gas['gas']

iterStep = 10
# KNN_scores = []
# for step in range(iterStep):

    # gas_train={}; gas_test={}
    # gas_train['gas'], gas_test['gas'], gas_train['label'], gas_test['label'] \
    #     = train_test_split(gas['sensor_data'], gas['labels'], test_size=0.2)
    # print(gas_train['label'].shape, gas_train['gas'].shape,
    #       gas_test['gas'].shape, gas_test['label'].shape)
    #
    # for i in range(len(gas_train['gas'])):
    #     gas_temp = gas_train['gas'][i]
    #     mean_value = np.mean(gas_temp, axis=0)
    #     std_value = np.std(gas_temp, axis=0)
    #     # print(mean_value.shape, std_value.shape)
    #     for j in range(len(mean_value)):
    #         gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]
    # # print(gas_train['gas'][3].shape)
    #
   ###The normalization of test dataset
for i in range(len(Gas['data'])):
    gas_temp1 = Gas['data'][i]
    gas_mean1 = np.mean(gas_temp1, axis=0)
    # print(gas_mean1)
    gas_std = np.std(gas_temp1, axis=0)
    for j in range(len(gas_mean1)):
        gas_temp1[:, j] = (gas_temp1[:, j] - gas_mean1[j]) / gas_std[j]


# def ConcatData(gasdata):
#     aX0, aX1, aX2 = np.shape(gasdata.transpose(0, 2, 1))
#     gasDataRe = []
#     gasdata = gasdata.transpose(0, 2, 1)
#     for i in range(aX0):
#         gasMatrix = gasdata[i]
#         gastemp = gasMatrix[:, 0][::10]
#         for j in range(aX2 - 1):
#             gastemp = np.concatenate((gastemp, gasMatrix[:, j + 1][::10]))
#         # print(gastemp.shape)
#         gasDataRe.append(gastemp)
#     return gasDataRe

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
# gasDataReTrain = np.array(ConcatData(gas_train['gas']))
# gasDataReTest = np.array(ConcatData(gas_test['gas']))
# gasLabelTrain = gas_train['label']
# gasLabelTest = gas_test['label']

# gasData = np.array(ConcatData(Gas['data']))
gasData = np.array(Gas['data'])
gasData = np.reshape(gasData, [593, -1])
print(np.shape(gasData))
gasLabel = reSetLabel(Gas)
# print((gasLabel))

pca = PCA(n_components=300)
pca.fit(gasData)
gasData_new = pca.transform(gasData)
print(gasData_new.shape)

# X_train, X_test, y_train, y_test = train_test_split(gasData_new, gasLabel, test_size=0.2,random_state=10)
# ###KNN分类器
knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_train)
# y_score = knn_clf.score(X_test, y_test)
# print(y_score)
knn_score = cross_val_score(knn_clf, gasData_new, gasLabel, cv=10)
knn_score_mean = np.mean(knn_score)
#     knn_score_mean = np.mean(knn_score)
#     knn_clf.fit(gasDataReTrain, gasLabelTrain)
#     knn_score = knn_clf.score(gasDataReTest, gasLabelTest)
#     print('Step#%d, the KNN score=%.6f'%(step, knn_score_mean))
#     KNN_scores.append(knn_score)
# KNN_scores_mean = np.mean(KNN_scores)
# print('\nAfter %d iteration:'%iterStep)
# print('the mean KNN score = %.6f' % KNN_scores_mean)

print('\n10 hold cross validation, the mean KNN score=%.5f' % knn_score_mean)

