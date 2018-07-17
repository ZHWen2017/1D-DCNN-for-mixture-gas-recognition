import numpy as np
# from MLPwithBN import *
import pickle
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

Max_steps = 2000
gas_path = '/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/gas_100.txt'

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
gas_data['gas'] = np.reshape(np.array(gas_data['gas']), [582, -1])
pca = PCA(n_components=20)
pca.fit(gas_data['gas'])
gasData = np.array(pca.transform(gas_data['gas']))
gasLabel = np.array(gas_data['label_'])

X_train, X_test, y_train, y_test = train_test_split(gasData, gasLabel)
labels = list(set())
# print('       10 hold Cross validating .....')
# for train_index, test_index in KF.split(gasData['label']):
#     train_set = {'gas':[], 'label':[]}; test_set = {'gas':[], 'label':[]}
#     for index0 in train_index:
#         train_set['gas'].append(gasData['gas'][index0])
#         train_set['label'].append(gasData['label'][index0])
#     for index1 in test_index:
#         test_set['gas'].append(gasData['gas'][index1])
#         test_set['label'].append(gasData['label'][index1])
#     train_set['gas'] = np.array(train_set['gas']); train_set['label'] = np.array(train_set['label'])
#     test_set['gas'] = np.array(test_set['gas']) ; test_set['label'] = np.array(test_set['label'])
#
#     print(np.shape(train_set['gas']), np.shape(train_set['label']),
#           np.shape(test_set['gas']), np.shape(test_set['label']))
#     _, length = np.shape(train_set['gas'])
#     # n_classes = np.unique(train_set['label'])
#     ###build MLP classifier
#     feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(train_set['gas'])
#     dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=6,
#                                              feature_columns=feature_columns)
#     dnn_clf.fit(x=train_set['gas'], y=train_set['label'], batch_size=64, steps=2000)
#
#     y_pred = list(dnn_clf.predict(test_set['gas']))
#     score = accuracy_score(test_set['label'], y_pred)
#     test_accs.append(score)
#
#
#     # test_acc, best_acc = MLP(hold_i, train_set, test_set, Max_steps, length)
#     # test_accs.append(best_acc)
#     # hold_i+=1
#     # print('the best accuracy = %.6f' % best_acc)
# acc_mean = np.mean(test_accs)
# print('\n\nAfter 10 fold Cross validation , the MLP mean accuracy: %.6f'% acc_mean)
