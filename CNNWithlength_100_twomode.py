import pickle
import numpy as np
import random
from sklearn.model_selection import KFold, train_test_split
from CNNWithBN_100set_single import *
from CNNWithlength_100_reload import *
from CNNWithBN_100set_onehot import *
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

Max_steps = 100
iterations = 20
# gas_path = '/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/gas_100.txt'
gas_path = '/home/wzh/PycharmProjects/data/multilabel/dataPlot/gas_100.txt'



##multi-label process
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
    = train_test_split(gasData['gas'], gasData['label'], test_size=0.2, random_state=7)

gas_train['label_one'] = reSetLabel(gas_train); gas_test['label_one'] = reSetLabel(gas_test)
gas_train['label_one'] = np.array(gas_train['label_one']);gas_test['label_one'] = np.array(gas_test['label_one'])
# gas_test['gas'], gas_valid['gas'], gas_test['label'], gas_valid['label'] = train_test_split(
#     gas_test['gas'], gas_test['label'], test_size=0.5, random_state=7
# )
# print(gas_train['gas'].shape, gas_train['label_one'].shape,gas_train['label'].shape, gas_test['gas'].shape, gas_test['label_one'].shape,
#       gas_test['label'].shape)
# acc_valid, acc_best, test_loss_best, train_loss, valid_loss, train_accs, times_mu = CNN_process(1, Max_steps, gas_train, gas_test,64)
# acc_valid_one, acc_best_one, test_loss_best_one, train_loss_one, valid_loss_one, train_accs_one, times_one = CNN_process_one(1, Max_steps, gas_train, gas_test, 64)
hidden_nums=[32, 64, 100, 128, 150, 180, 200, 250, 300, 350, 400, 450, 512]
times_multi=[]; times_onehot=[]
for hidden_num in hidden_nums:
    print("Hidden number is %d ..."%hidden_num)
    acc_valid, acc_best, test_loss_best, train_loss, valid_loss, train_accs, times_mu = CNN_process(1, 100, gas_train, gas_test, hidden_num)
    acc_valid_one, acc_best_one, test_loss_best_one, train_loss_one, valid_loss_one, train_accs_one, times_one = CNN_process_one(1, 100, gas_train, gas_test, hidden_num)
    times_multi.append(times_mu)
    times_onehot.append(times_one)
print("Multi-label process: ", times_multi)
print("One hot process: ", times_onehot)
# acc_test, loss_test = CNN_process_reload(gas_test)

# test_result = {}
# test_result['acc']=acc_valid;test_result['acc_best']=acc_best;test_result['test_loss']=valid_loss
# test_result['train_loss'] = train_loss; test_result['test_loss_best'] = test_loss_best
# test_result['train_acc'] = train_accs
#
# test_path='/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/result/test_result_627.txt'
# f2 = open(test_path, 'wb')
# pickle.dump(test_result, f2)
# f2.close()




