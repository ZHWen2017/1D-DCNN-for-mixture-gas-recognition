import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

timescale = 10
methane_path='../../ethylene_methane-1.txt'
CO_path = '../../ethylene_CO-1.txt'
# x = np.loadtxt('ethylene_CO-1.txt')
name = ['time', 'me/co', 'eth', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16']
x0 = pd.read_table(methane_path, sep='\s+', names=name)
x1 = pd.read_table(CO_path, sep='\s+', names=name)
x0 = x0.values; x1 = x1.values
print(x0.shape, x1.shape)

sensor_data0 = x0[:, 3:]
odors_data0 = x0[:, 1:3]
times0 = x0[:, 0]
sensor_data1 = x1[:, 3:]
odors_data1 = x1[:, 1:3]
times1 = x1[:, 0]

sensor_data0 = sensor_data0[10000::timescale]
odors_data0 = odors_data0[10000::timescale]
times0 = times0[10000::timescale]
sensor_data1 = sensor_data1[10000::timescale]
odors_data1 = odors_data1[10000::timescale]
times1 = times1[10000::timescale]
print(sensor_data0.shape, odors_data0.shape, times0.shape,
      sensor_data1.shape, odors_data1.shape, times1.shape)

labelsMe = []
labelsCO = []
for data in odors_data0:
    if data[0] == 0 and data[1] != 0:
        labelsMe.append([1, 0, 0])
    if data[0] != 0 and data[1] == 0:
        labelsMe.append([0, 0, 1, ])
    if data[0] != 0 and data[1] != 0:
        labelsMe.append([1, 0, 1])
    if data[0] == 0 and data[1] == 0:
        labelsMe.append([0, 0, 0])

for data in odors_data1:
    if data[0] == 0 and data[1] != 0:
        labelsCO.append([1, 0, 0])
    if data[0] != 0 and data[1] == 0:
        labelsCO.append([0, 1,  0])
    if data[0] != 0 and data[1] != 0:
        labelsCO.append([1, 1, 0])
    if data[0] == 0 and data[1] == 0:
        labelsCO.append([0, 0, 0])
print(len(labelsMe), len(labelsCO))



def measure_label0(labels, label_temp):
    j = 0
    for label in labels:
        if (label == label_temp):
            j += 1
        else:
            break
    labels_new = labels[j:]
    return j ,labels_new



labels_split = labelsMe
labels_split1 = labelsCO
i = 0
j = 0
index_class_me = [0]
index_class_CO = [0]
while i < len(labelsMe):
    if (labelsMe[i] == labels_split[0]):
        labels_init = labelsMe[i]
        index_j, labels_split = measure_label0(labels_split, labels_init)
        i = index_j + i
        index_class_me.append(i)

while j < len(labelsCO):
    if (labelsCO[j] == labels_split1[0]):
        labels_init1 = labelsCO[j]
        index_CO, labels_split1 = measure_label0(labels_split1, labels_init1)
        j = index_CO + j
        index_class_CO.append(j)

print(len(index_class_me), len(index_class_CO))

eth_Me = {'gas':[], 'label':[], 'times':[]}; eth_CO = {'gas':[], 'label':[], 'times':[]}
for id in range(len(index_class_me)-1):
    eth_Me['gas'].append(sensor_data0[index_class_me[id]:index_class_me[id+1]])
    eth_Me['label'].append(labelsMe[index_class_me[id]:index_class_me[id+1]])
    eth_Me['times'].append(times0[index_class_me[id]:index_class_me[id+1]])
print('eth_Me length: ', np.shape(eth_Me['gas']), np.shape(eth_Me['label']), len(eth_Me['times']))
# print(eth_Me['label'][15])

for ie in range(len(index_class_CO) - 1):
    eth_CO['gas'].append(sensor_data1[index_class_CO[ie]:index_class_CO[ie+1]])
    eth_CO['label'].append(labelsCO[index_class_CO[ie]:index_class_CO[ie+1]])
    eth_CO['times'].append(times1[index_class_CO[ie]:index_class_CO[ie+1]])
print('eth_CO length: ', np.shape(eth_CO['gas']), np.shape(eth_CO['label']), len(eth_CO['times']))

#
length_CO = []
for i in range(len(eth_CO['label'])):
#    if eth_CO['label'][i][0] == [1, 1, 0]:
        X = eth_CO['times'][i]
        Y = eth_CO['gas'][i]
        label_i = eth_CO['label'][i][0]
        length_CO.append(len(label_i))

CO_plit_index = []
for i in range(len(eth_CO['label'])):
    if eth_CO['label'][i][0] == [1, 1, 0]:
        if eth_CO['times'][i][-1] - eth_CO['times'][i][0] > 200:
            X = eth_CO['times'][i]
            Y = eth_CO['gas'][i]
            label_i = eth_CO['label'][i][0]
            CO_plit_index.append(i)
print("CO split index is: %s" % CO_plit_index)

            
Me_plit_index = []
for i in range(len(eth_Me['label'])):
    if eth_Me['label'][i][0] == [1, 0, 1]:
        if eth_Me['times'][i][-1] - eth_Me['times'][i][0] > 200:
                
            X = eth_Me['times'][i]
            Y = eth_Me['gas'][i]
            label_i = eth_Me['label'][i][0]
            Me_plit_index.append(i)
# print("Me split index is %s" % Me_plit_index)
# print([len(eth_Me['label'][i]) for i in Me_plit_index])
# Me_plit_index = np.delete(Me_plit_index, [0, 6, 11, 13], axis=0)


print([eth_CO['label'][i][0] for i in CO_plit_index])



eth_CO_split={'gas':[],'label':[],'times':[]}
eth_Me_split={'gas':[],'label':[],'times':[]}

for index in CO_plit_index:
    gas_temp = eth_CO['gas'][index]
    index_s = len(gas_temp) // 2
    gas_temp0 = eth_CO['gas'][index][:index_s]; gas_temp1 = eth_CO['gas'][index][index_s:]
    label_temp0= eth_CO['label'][index][:index_s]; times_temp0 = eth_CO['times'][index][:index_s]
    label_temp1= eth_CO['label'][index][index_s:]; times_temp1 = eth_CO['times'][index][index_s:]
    eth_CO_split['gas'].append(gas_temp0);eth_CO_split['label'].append(label_temp0)
    eth_CO_split['times'].append(times_temp0)
    eth_CO_split['gas'].append(gas_temp1);eth_CO_split['label'].append(label_temp1)
    eth_CO_split['times'].append(times_temp1)

for index_m in Me_plit_index:
    index_ms = len(eth_Me['gas'][index_m]) // 2
    
    gas_temp_m0 = eth_Me['gas'][index_m][:index_ms]; gas_temp_m1 = eth_Me['gas'][index_m][index_ms:]
    label_temp_m0 = eth_Me['label'][index_m][:index_ms]; times_temp_m0 = eth_Me['times'][index][:index_ms]
    label_temp_m1 = eth_Me['label'][index_m][index_ms:]; times_temp_m1 = eth_Me['times'][index][index_ms:]
    
    eth_Me_split['gas'].append(gas_temp_m0); eth_Me_split['label'].append(label_temp_m0)
    eth_Me_split['times'].append(times_temp_m0)
    eth_Me_split['gas'].append(gas_temp_m1); eth_Me_split['label'].append(label_temp_m1)
    eth_Me_split['times'].append(times_temp_m1)

eth_CO['gas'] = np.delete(eth_CO['gas'], CO_plit_index, axis=0)
eth_CO['label'] = np.delete(eth_CO['label'], CO_plit_index, axis=0)
eth_CO['times'] = np.delete(eth_CO['times'], CO_plit_index, axis=0)
eth_Me['gas'] = np.delete(eth_Me['gas'], Me_plit_index, axis=0)
eth_Me['label'] = np.delete(eth_Me['label'], Me_plit_index, axis=0)
eth_Me['times'] = np.delete(eth_Me['times'], Me_plit_index, axis=0)
print([eth_CO['label'][i][0] for i in CO_plit_index])

gas_final = {'gas':[], 'label':[],'times':[]}
for n0 in range(len(eth_CO['label'])):
    gas_final['gas'].append(eth_CO['gas'][n0])
    gas_final['label'].append(eth_CO['label'][n0])
    gas_final['times'].append(eth_CO['times'][n0])

for n1 in range(len(eth_CO_split['label'])):
    gas_final['gas'].append(eth_CO_split['gas'][n1])
    gas_final['label'].append(eth_CO_split['label'][n1])
    gas_final['times'].append(eth_CO_split['times'][n1])
    # print(eth_CO_split['label'][n1])

for n2 in range(len(eth_Me['label'])):
    gas_final['gas'].append(eth_Me['gas'][n2])
    gas_final['label'].append(eth_Me['label'][n2])
    gas_final['times'].append(eth_Me['times'][n2])

for n3 in range(len(eth_Me_split['label'])):
    gas_final['gas'].append(eth_Me_split['gas'][n3])
    gas_final['label'].append(eth_Me_split['label'][n3])
    gas_final['times'].append(eth_Me_split['times'][n3])
print('the final gas shape: ', np.shape(gas_final['gas']), np.shape(gas_final['label']))

less_100 = [i for i in range(len(gas_final['label'])) if len(gas_final['label'][i]) < 100]
gas_final['gas'] = (list(np.delete(gas_final['gas'], less_100, axis=0)))
gas_final['label'] = (list(np.delete(gas_final['label'], less_100, axis=0)))
gas_final['times'] = list(np.delete(gas_final['times'], less_100, axis=0))
minestLength = len(gas_final['gas'][0])
index_100 = []
for i in range(len(gas_final['gas'])):
    if len(gas_final['gas'][i]) == 100:
        index_100.append(i)
    if len(gas_final['gas'][i]) < minestLength:
        minestLength = len(gas_final['gas'][i])
print('the minest length = %d' % minestLength)
print('the index of 100 length: ', index_100)
grad_gas = []; gas_process={'gas':[], 'label':[]}
for n4 in range(len(gas_final['label'])):
    index_sort = np.arange(len(gas_final['label'][n4]) - minestLength)
    gas_bigest = 0
    for index_1 in index_sort:
        index_end = index_1 + minestLength
        gas_def1 = np.absolute(gas_final['gas'][n4][index_1][9] - gas_final['gas'][n4][index_end][9])
        gas_dif1 = (np.divide(gas_def1, gas_final['gas'][n4][index_1][9]))
        if gas_dif1 > gas_bigest:
            gas_bigest = gas_dif1
    grad_gas.append(gas_bigest)
    
    for index_2 in index_sort:
        index_2_end = index_2 + minestLength
        gas_def2 = np.absolute(gas_final['gas'][n4][index_2][9] - gas_final['gas'][n4][index_2_end][9])
        gas_dif2 = (np.divide(gas_def2, gas_final['gas'][n4][index_2][9]))
        if gas_dif2 == gas_bigest:
            gas_process['gas'].append(gas_final['gas'][n4][index_2:index_2_end])
            gas_process['label'].append(gas_final['label'][n4][0])
for index in index_100:
    gas_process['gas'].append(gas_final['gas'][index])
    gas_process['label'].append(gas_final['label'][index][0])
print(np.shape(gas_process['gas']), np.shape(gas_process['label']))
gas_process['gas'] = np.transpose(gas_process['gas'], (0, 2, 1))

index_r0 = np.arange(len(gas_process['gas'][:294]))
index_r1 = np.arange(len(gas_process['label'][294:])) + 294
random.shuffle(index_r0)
random.shuffle(index_r1)
split_index0 = int(len(index_r0) * 0.8)
split_index1 = int(len(index_r1) * 0.8)
train_index = index_r0[:split_index0]; test_index = index_r0[split_index0:]
train_index = np.concatenate((train_index, index_r1[:split_index1]))
test_index = np.concatenate((test_index, index_r1[split_index1:]))

train_set = {'gas':[], 'label':[]}
test_set = {'gas':[], 'label':[]}
for t0 in train_index:
    train_set['gas'].append(gas_process['gas'][t0])
    train_set['label'].append(gas_process['label'][t0])

for t1 in test_index:
    test_set['gas'].append(gas_process['gas'][t1])
    test_set['label'].append(gas_process['label'][t1])

train_path = './train_set.txt'
test_path = './test_set.txt'
gas_path = './gas_100.txt'
gasraw_path ='./gasraw.txt'
#f1 = open(train_path, 'wb')
#pickle.dump(train_set, f1)
#f1.close()
#
#f2 = open(test_path, 'wb')
#pickle.dump(test_set, f2)
#f1.close()
f3 = open(gas_path, 'wb')
pickle.dump(gas_process, f3)
f3.close()
#
# f4 = open(gasraw_path, 'wb')
# pickle.dump(gas_final, f4)
# f4.close()


#            fig1 = plt.figure()
#            ax4 = fig1.add_subplot(111)
#        
#            ax4.plot(X, Y[:, 0], 'r--', linewidth=1)
#            ax4.plot(X, Y[:, 1], 'o--', linewidth=1)
#            ax4.plot(X, Y[:, 2], 'y--', linewidth=1)
#            ax4.plot(X, Y[:, 3], 'g--', linewidth=1)
#            ax4.plot(X, Y[:, 4], 'k--', linewidth=1)
#            ax4.plot(X, Y[:, 5], 'b--', linewidth=1)
#            ax4.plot(X, Y[:, 6], 'p--', linewidth=1)
#            ax4.plot(X, Y[:, 7], 'b--', linewidth=1)
#            ax4.plot(X, Y[:, 8], 'r-.', linewidth=1)
#            ax4.plot(X, Y[:, 9], 'o-.', linewidth=1)
#            ax4.plot(X, Y[:, 10], 'y-.', linewidth=1)
#            ax4.plot(X, Y[:, 11], 'g-.', linewidth=1)
#            ax4.plot(X, Y[:, 12], 'k-.', linewidth=1)
#            ax4.plot(X, Y[:, 13], 'b-.', linewidth=1)
#            ax4.plot(X, Y[:, 14], 'p-.', linewidth=1)
#            ax4.plot(X, Y[:, 15], 'b-.', linewidth=1)
#            label_name = str(list(label_i))
#            label_name = label_name + '  '+'%d' % i
#            ax4.set_xlabel(label_name)
#            save_path= './picture2/Fig1_%d.eps' % i
#            plt.savefig(save_path)
#plt.show()
# print(eth_CO['label'][15])

#n_items = len(eth_CO['gas'])
#for i in range(n_items):
#    X = eth_CO['times'][i]
#    Y = eth_CO['gas'][i]
#    label_i = eth_CO['label'][i][0]
#    fig = plt.figure()
#    ax3 = fig.add_subplot(111)
#
##    ax3.plot(X, Y[:, 0], 'r--', linewidth=1)
##    ax3.plot(X, Y[:, 1], 'o--', linewidth=1)
##    ax3.plot(X, Y[:, 2], 'y--', linewidth=1)
##    ax3.plot(X, Y[:, 3], 'g--', linewidth=1)
##    ax3.plot(X, Y[:, 4], 'k--', linewidth=1)
##    ax3.plot(X, Y[:, 5], 'b--', linewidth=1)
##    ax3.plot(X, Y[:, 6], 'p--', linewidth=1)
##    ax3.plot(X, Y[:, 7], 'b--', linewidth=1)
#    ax3.plot(X, Y[:, 8], 'r-.', linewidth=1)
#    ax3.plot(X, Y[:, 9], 'o-.', linewidth=1)
#    ax3.plot(X, Y[:, 10], 'y-.', linewidth=1)
#    ax3.plot(X, Y[:, 11], 'g-.', linewidth=1)
#    ax3.plot(X, Y[:, 12], 'k-.', linewidth=1)
#    ax3.plot(X, Y[:, 13], 'b-.', linewidth=1)
#    ax3.plot(X, Y[:, 14], 'p-.', linewidth=1)
#    ax3.plot(X, Y[:, 15], 'b-.', linewidth=1)
#    label_name = str(list(label_i))
#    ax3.set_xlabel(label_name)
##    name = './picture/fig_%d.eps' % i
##    plt.savefig(name)
#plt.show()
#
#n_items1 = len(eth_Me['gas'])
#for i in range(n_items1):
#    X = eth_Me['times'][i]
#    Y = eth_Me['gas'][i]
#    label_i = eth_Me['label'][i][0]
#    fig = plt.figure()
#    ax3 = fig.add_subplot(111)
#
#    ax3.plot(X, Y[:, 0], 'r--', linewidth=1)
#    ax3.plot(X, Y[:, 1], 'o--', linewidth=1)
#    ax3.plot(X, Y[:, 2], 'y--', linewidth=1)
#    ax3.plot(X, Y[:, 3], 'g--', linewidth=1)
#    ax3.plot(X, Y[:, 4], 'k--', linewidth=1)
#    ax3.plot(X, Y[:, 5], 'b--', linewidth=1)
#    ax3.plot(X, Y[:, 6], 'p--', linewidth=1)
#    ax3.plot(X, Y[:, 7], 'b--', linewidth=1)
#    ax3.plot(X, Y[:, 8], 'r-.', linewidth=1)
#    ax3.plot(X, Y[:, 9], 'o-.', linewidth=1)
#    ax3.plot(X, Y[:, 10], 'y-.', linewidth=1)
#    ax3.plot(X, Y[:, 11], 'g-.', linewidth=1)
#    ax3.plot(X, Y[:, 12], 'k-.', linewidth=1)
#    ax3.plot(X, Y[:, 13], 'b-.', linewidth=1)
#    ax3.plot(X, Y[:, 14], 'p-.', linewidth=1)
#    ax3.plot(X, Y[:, 15], 'b-.', linewidth=1)
#    label_name = str(list(label_i))
#    ax3.set_xlabel(label_name)
#    name = './picture1/fig_%d.eps' % i
#    plt.savefig(name)
#plt.show()