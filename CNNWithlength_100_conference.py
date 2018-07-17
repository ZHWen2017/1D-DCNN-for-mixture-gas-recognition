import pickle
import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split
from CNNWithBN_100set_cvd import *
from CNNWithlength_100_reload import *

Max_steps = 2000
iterations = 20
# gas_path = '/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/gas_100.txt'
gas_path = '/home/wzh/PycharmProjects/data/multilabel/dataPlot/gas_conference.txt'

f1 = open(gas_path, 'rb')
gas_data = pickle.load(f1)
f1.close()
print(gas_data['gas'].shape, np.shape(gas_data['label']))

gas_index = np.arange(len(gas_data['label']))
np.random.seed(2018)
np.random.shuffle(gas_index)
gasData = {'gas':[], 'label':[]}
for i in gas_index:
    gasData['gas'].append(gas_data['gas'][i])
    gasData['label'].append(gas_data['label'][i])

gasData['gas'] = np.array(gasData['gas']); gasData['label'] = np.array(gasData['label'])
gasData['gas'] = np.reshape(gasData['gas'], np.shape(gasData['gas']) + (1,))
gas_train={}; gas_test={}; gas_valid = {}
gas_train['gas'], gas_test['gas'], gas_train['label'], gas_test['label'] \
    = train_test_split(gasData['gas'], gasData['label'], test_size=0.2, random_state=8)
# gas_test['gas'], gas_valid['gas'], gas_test['label'], gas_valid['label'] = train_test_split(
#     gas_test['gas'], gas_test['label'], test_size=0.5, random_state=7
# )
# print(gas_train['gas'].shape, gas_train['label'].shape, gas_test['gas'].shape, gas_test['label'].shape)
acc_valid, acc_best, test_loss_best, train_loss, valid_loss, train_accs = CNN_process(1, Max_steps, gas_train, gas_test)
# acc_test, loss_test = CNN_process_reload(gas_test)

test_result = {}
test_result['acc']=acc_valid;test_result['acc_best']=acc_best;test_result['test_loss']=valid_loss
test_result['train_loss'] = train_loss; test_result['test_loss_best'] = test_loss_best
test_result['train_acc'] = train_accs

test_path='/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/result/test_result_conference.txt'
f2 = open(test_path, 'wb')
pickle.dump(test_result, f2)
f2.close()