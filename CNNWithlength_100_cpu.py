import pickle
import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split
from CNNWithBN_100set_cvd import *

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

Max_steps = 2000
iterations = 20
gas_path = '/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/gas_100.txt'
# gas_path = 'C:/Users/Administrator/Desktop/resultPlot/dataPlot/gas_100.txt'
f1 = open(gas_path, 'rb')
gas_data = pickle.load(f1)
f1.close()
# print(gas_data['gas'].shape, np.shape(gas_data['label']))
for i in range(len(gas_data['gas'])):
    gas_temp = gas_data['gas'][i]
    mean_value = np.mean(gas_temp, axis=0)
    std_value = np.std(gas_temp, axis=0)
    # print(mean_value.shape, std_value.shape)
    for j in range(len(mean_value)):
        gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]

gasData = {'gas':[], 'label':[]}
gasData['gas'] = gas_data['gas']; gasData['label'] = gas_data['label']
gasData['gas'] = np.array(gasData['gas']); gasData['label'] = np.array(gasData['label'])
gasData['gas'] = np.reshape(gasData['gas'], np.shape(gasData['gas']) + (1,))
gas_train={}; gas_test={}
gas_train['gas'], gas_test['gas'], gas_train['label'], gas_test['label'] \
    = train_test_split(gasData['gas'], gasData['label'], test_size=0.2, random_state=10)
y_test = reSetLabel(gas_test)
# print(gas_train['gas'].shape, gas_train['label'].shape, gas_test['gas'].shape, gas_test['label'].shape)
acc_valid, acc_best, test_loss_best, train_loss, test_loss, train_accs, y_predict = CNN_process(1, Max_steps, gas_train, gas_test)

test_result = {}
test_result['acc']=acc_valid;test_result['acc_best']=acc_best;test_result['test_loss']=test_loss
test_result['train_loss'] = train_loss; test_result['test_loss_best'] = test_loss_best
test_result['train_acc'] = train_accs
test_predict = {}
test_predict['1D-DCNN'] = np.array([y_test,y_predict])
test_path='./test_result_1D-DCNN.txt'
f1 = open(test_path, 'wb')
pickle.dump(test_predict, f1)
f1.close()
# test_path='/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/result/test_result.txt'
# f2 = open(test_path, 'wb')
# pickle.dump(test_result, f2)
# f2.close()