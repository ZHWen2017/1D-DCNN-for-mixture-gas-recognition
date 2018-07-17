import pickle
import numpy as np
import random
from sklearn.model_selection import KFold
from CNNWithBN_100set_cv import *

Max_steps = 3000
iterations = 20
gas_path = '/home/wzh/PycharmProjects/data/multilabel/dataPlot/gas_conference.txt'


f1 = open(gas_path, 'rb')
gas_data = pickle.load(f1)
f1.close()
print(gas_data['gas'].shape, np.shape(gas_data['label']))

gas_index = np.arange(len(gas_data['label']))
np.random.seed(2)
np.random.shuffle(gas_index)
gasData = {'gas':[], 'label':[]}
for i in gas_index:
    gasData['gas'].append(gas_data['gas'][i])
    gasData['label'].append(gas_data['label'][i])




KF = KFold(n_splits=10, random_state=2018)
hold_i = 1
test_accs = []; loss_bests = []; accs = []
print('       10 hold Cross validating .....')
for train_index, test_index in KF.split(gasData['label']):

    train_set = {'gas':[], 'label':[]}; test_set = {'gas':[], 'label':[]}
    for index0 in train_index:
        train_set['gas'].append(gasData['gas'][index0])
        train_set['label'].append(gasData['label'][index0])
    for index1 in test_index:
        test_set['gas'].append(gasData['gas'][index1])
        test_set['label'].append(gasData['label'][index1])
    train_set['gas'] = np.array(train_set['gas']); train_set['label'] = np.array(train_set['label'])
    test_set['gas'] = np.array(test_set['gas']) ; test_set['label'] = np.array(test_set['label'])
    print(np.shape(train_set['gas']), np.shape(train_set['label']),
          np.shape(test_set['gas']), np.shape(test_set['label']))

    gas_train = train_set; gas_test = test_set
    gas_train['gas'] = np.reshape(gas_train['gas'], gas_train['gas'].shape + (1,))
    gas_test['gas'] = np.reshape(gas_test['gas'], gas_test['gas'].shape + (1,))
    for i in range(len(gas_train['gas'])):
        gas_temp = gas_train['gas'][i]
        mean_value = np.mean(gas_temp, axis=0)
        std_value = np.std(gas_temp, axis=0)
        # print(mean_value.shape, std_value.shape)
        for j in range(len(mean_value)):
            gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]
    # print(gas_train['gas'][3].shape)

    ###The normalization of test dataset
    for i in range(len(gas_test['gas'])):
        gas_temp1 = gas_test['gas'][i]
        gas_mean = np.mean(gas_temp1, axis=0)
        gas_std = np.std(gas_temp1, axis=0)
        for j in range(len(gas_mean)):
            gas_temp1[:, j] = (gas_temp1[:, j] - gas_mean[j]) / gas_std[j]
    print('---------------The %d Hold validation------------' % hold_i)
    acc_best, loss_best = CNN_process(hold_i, Max_steps, train_set, test_set)
    test_accs.append(acc_best); loss_bests.append(loss_best)
    hold_i += 1
test_accs_mean = np.mean(test_accs)
accs.append(test_accs_mean)
print(accs)
# print('\nAfter %d iteration, 10 hold cross validation: '% iterations)
print('                                 The mean acc value=%.7f' % test_accs_mean)