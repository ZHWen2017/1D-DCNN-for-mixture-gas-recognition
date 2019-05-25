from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from  sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

f = open("gas.txt", "rb")
gas0 = pickle.load(f)
# gas_test['label'] = np.array(gas_test['label'])
print(gas0['labels'].shape, gas0['sensor_data'].shape)
f.close()

gas = {}
gas['label'] = gas0['labels']; gas['data'] = gas0['sensor_data']

iterStep = 10
rnd_scores=[]
for step in range(iterStep):

    for i in range(len(gas['label'])):
        gas_temp = gas['data'][i]
        mean_value = np.mean(gas_temp, axis=0)
        std_value = np.std(gas_temp, axis=0)
        # print(mean_value.shape, std_value.shape)
        for j in range(len(mean_value)):
            gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]
    def reSetLabel(gas_train):
        n0, m0 = np.shape(gas_train['label'])
        gas_train['label_co'] = []
        for i in range(n0):
            if (gas_train['label'][i][0] == 1 and gas_train['label'][i][1] == 0
                    and gas_train['label'][i][2] == 0):
                gas_train['label_co'].append(0)
                # print('append 1: ', i)
            elif (gas_train['label'][i][0] == 0 and gas_train['label'][i][1] == 1
                    and gas_train['label'][i][2] == 0):
                gas_train['label_co'].append(1)
                # print('append 2: ', i)
            elif (gas_train['label'][i][0] == 0 and gas_train['label'][i][1] == 0
                    and gas_train['label'][i][2] == 1):
                gas_train['label_co'].append (2)
                # print('append 3: ', i)
            elif (gas_train['label'][i][0] == 1 and gas_train['label'][i][1] == 1
                    and gas_train['label'][i][2] == 0):
                gas_train['label_co'].append(3)
                # print('append 4: ', i)
            elif (gas_train['label'][i][0] == 1 and gas_train['label'][i][1] == 0
                    and gas_train['label'][i][2] == 1):
                gas_train['label_co'].append(4)
                # print('append 5: ', i)
            else:
                print('append emty')
        return gas_train['label_co']

    gas['label_co'] = reSetLabel(gas)
    # print(len(gas_train['label_co']), len(gas_train['label']))
    # print(len(gas_test['label_co']), len(gas_test['label']))

    def ConcatData(gasdata):
        aX0, aX1, aX2 = np.shape(gasdata.transpose(0, 2, 1))
        gasDataRe = []
        gasdata = gasdata.transpose(0, 2, 1)
        for i in range(aX0):
            gasMatrix = gasdata[i]
            gastemp = gasMatrix[:, 0][::2]
            for j in range(aX2-1):
                gastemp = np.concatenate((gastemp, gasMatrix[:, j+1][::2]))
            # print(gastemp.shape)
            gasDataRe.append(gastemp)
        return gasDataRe

    gasDataRe = np.array(ConcatData(gas['data']))
    gasLabel = label_binarize(gas['label_co'], classes=list(range(5)))
    # print(gasLabel.shape, gasDataRe.shape)

    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()


    rnd_score = cross_val_score(rnd_clf, gasDataRe, gasLabel, cv=10)
    rnd_score_mean = np.mean(rnd_score)
    rnd_scores.append(rnd_score_mean)
rnd_score_means = np.mean(rnd_scores)
print('After 10 hold Cross Validation, the accuracy of RandomForest is: %.5f' % rnd_score_means)

