import pickle
import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split
from CNNWithBN_100set_cvd import *
from CNNWithlength_100_reload import *

Max_steps = 2000
iterations = 20
# gas_path = '/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/gas_100.txt'
gas_path = '/home/wzh/PycharmProjects/data/multilabel/dataPlot/gas_100.txt'

f1 = open(gas_path, 'rb')
gas_data = pickle.load(f1)
f1.close()
print(gas_data['gas'].shape, np.shape(gas_data['label']))


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

gas_index = np.arange(len(gas_data['label']))
np.random.seed(15)
np.random.shuffle(gas_index)
gasData = {'gas':[], 'label':[]}
for i in gas_index:
    gasData['gas'].append(gas_data['gas'][i])
    gasData['label'].append(gas_data['label'][i])

gasData['gas'] = np.array(gasData['gas']); gasData['label'] = np.array(gasData['label'])
gasData['gas'] = np.reshape(gasData['gas'], np.shape(gasData['gas']) + (1,))
gas_train={}; gas_test={}; gas_valid = {}
gas_train['gas'], gas_test['gas'], gas_train['label'], gas_test['label'] \
    = train_test_split(gasData['gas'], gasData['label'], test_size=0.2, random_state=7)
# gas_test['gas'], gas_valid['gas'], gas_test['label'], gas_valid['label'] = train_test_split(
#     gas_test['gas'], gas_test['label'], test_size=0.5, random_state=7
# )
# print(gas_train['gas'].shape, gas_train['label'].shape, gas_test['gas'].shape, gas_test['label'].shape)
# acc_valid, acc_best, test_loss_best, train_loss, valid_loss, train_accs = CNN_process(1, Max_steps, gas_train, gas_test)
acc_test, loss_test, final_feature = CNN_process_reload(gasData)
print('final_feature shape: ', final_feature.shape)

gas_label = reSetLabel(gasData)


test_result = {}
test_result['acc_whole'] = acc_test
test_result['loss_whole'] = loss_test
test_result['final_feature'] = final_feature
test_result['label'] = gas_label

test_path='/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/result/whole_set.txt'
f2 = open(test_path, 'wb')
pickle.dump(test_result, f2)
f2.close()