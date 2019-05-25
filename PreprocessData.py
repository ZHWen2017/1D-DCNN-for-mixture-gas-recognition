import numpy as np
import pickle


def processGas():
    ##The data preprocessing\
    gas = {}
    f1 = open("../../gas.txt", "rb")
    gas1 = pickle.load(f1)
    gas['gas'] = gas1['sensor_data']
    gas['label'] = gas1['labels']
    # gas_train['label'] = np.array(gas_train['label'])
    f1.close()



    print(gas['gas'].shape)
    print(gas['label'].shape)
      ##The normalization of training dataset
    for i in range(len(gas['gas'])):
        gas_temp = gas['gas'][i]
        mean_value = np.mean(gas_temp, axis=0)
        std_value = np.std(gas_temp, axis=0)
        # print(mean_value.shape, std_value.shape)
        for j in range(len(mean_value)):
            gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]
    # print(gas_train['gas'][3].shape)


    gas['gas'] = np.reshape(gas['gas'], gas['gas'].shape+(1,))
    print(gas['gas'].shape)

    ###split gas data
    gas_spilt = {}
    gas_spilt['gas'] = np.split(gas['gas'][:630], 10, axis=0)
    gas_spilt['labels'] = np.split(gas['label'][:630], 10, axis=0)
    print(len(gas_spilt['gas'][9]), len(gas_spilt['labels'][9]))
    valid_set = {'gas':[], 'labels':[]}
    train_set = {'gas':[], 'labels':[]}
    for i in range(10):
        valid_set['gas'].append(gas_spilt['gas'][i])
        valid_set['labels'].append(gas_spilt['labels'][i])
        gas_delete = np.delete(gas_spilt['gas'], i, 0)
        labels_delete = np.delete(gas_spilt['labels'], i, 0)
        gas_stack = np.concatenate((gas_delete[0],gas_delete[1],gas_delete[2],gas_delete[3],
                                    gas_delete[4],gas_delete[5], gas_delete[6], gas_delete[7], gas_delete[8]), axis=0)
        labels_stack = np.concatenate((labels_delete[0], labels_delete[1], labels_delete[2],
                                       labels_delete[3], labels_delete[4], labels_delete[5],
                                       labels_delete[6], labels_delete[7], labels_delete[8]), axis=0)
        train_set['gas'].append(gas_stack)
        train_set['labels'].append(labels_stack)
    valid_set['gas'] = np.array(valid_set['gas'])
    valid_set['labels'] = np.array(valid_set['labels'])
    train_set['gas'] = np.array(train_set['gas'])
    train_set['labels'] = np.array(train_set['labels'])
    print(valid_set['gas'].shape, valid_set['labels'].shape, train_set['gas'].shape, train_set['labels'].shape)
    return train_set, valid_set