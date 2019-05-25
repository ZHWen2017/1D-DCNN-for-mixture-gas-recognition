import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from scipy import interp
length = 100

f = open("C:/Users/Administrator/Desktop/resultPlot/dataPlot/gas_100.txt", "rb")
gas = pickle.load(f)
# gas_test['label'] = np.array(gas_test['label'])
print(np.shape(gas['label']), np.shape(gas['gas']))
f.close()

Gas = {}
Gas['label'] = gas['label']
Gas['data'] = gas['gas']

for i in range(len(Gas['data'])):
    gas_temp1 = Gas['data'][i]
    gas_mean1 = np.mean(gas_temp1, axis=0)
    # print(gas_mean1)
    gas_std = np.std(gas_temp1, axis=0)
    for j in range(len(gas_mean1)):
        gas_temp1[:, j] = (gas_temp1[:, j] - gas_mean1[j]) / gas_std[j]

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
    # gas_train['label_co'] = reSetLabel(gas_train)
    # gas_test['label_co'] = reSetLabel(gas_test)
    # print(len(gas_train['label_co']), len(gas_train['label']))
    # print(len(gas_test['label_co']), len(gas_test['label']))
Gas['label_co'] = reSetLabel(Gas)

def ConcatData(gasdata):
    aX0, aX1, aX2 = np.shape(gasdata.transpose(0, 2, 1))
    gasDataRe = []
    gasdata = gasdata.transpose(0, 2, 1)
    for i in range(aX0):
        gasMatrix = gasdata[i]
        gastemp = gasMatrix[:, 0][::10]
        for j in range(aX2-1):
            gastemp = np.concatenate((gastemp, gasMatrix[:, j+1][::10]))
        # print(gastemp.shape)
        gasDataRe.append(gastemp)
    return gasDataRe

    # gasDataReTrain = np.array(ConcatData(gas_train['gas']))
    # gasDataReTest = np.array(ConcatData(gas_test['gas']))
    # gasLabelTrain = label_binarize(gas_train['label_co'], classes=list(range(5)))
    # gasLabelTest = label_binarize(gas_test['label_co'], classes=list(range(5)))
    # print(gasLabelTrain.shape, gasDataReTrain.shape)

# gasDataRe = np.array(ConcatData(Gas['data']))
gasDataRe = np.reshape(np.array(Gas['data']),[582,-1])
pca = PCA(n_components=10)
pca.fit(gasDataRe)
gasDataRe = np.array(pca.transform(gasDataRe))
gasLabel = np.array(Gas['label_co'])
cv = StratifiedKFold(n_splits=10, shuffle=False)
classifier_svm = svm.SVC(kernel='linear', probability=True)
classifier_rf = RandomForestClassifier(n_estimators=500)

tprs_svm = []; aucs_svm = []; i = 0
rss_rf = []; rss_svm = []
mean_fpr = np.linspace(0, 1, 100)
for train_index, test_index in cv.split(gasDataRe, gasLabel):
    predict_svm = classifier_svm.fit(gasDataRe[train_index], gasLabel[train_index]).predict(gasDataRe[test_index])
    predict_rf = classifier_rf.fit(gasDataRe[train_index], gasLabel[train_index]).predict(gasDataRe[test_index])
    rs_svm = recall_score(gasLabel[test_index], predict_svm, average='weighted')
    rs_rf = recall_score(gasLabel[test_index], predict_rf, average='weighted')
    precision_svm = precision_score(gasLabel[test_index], predict_svm, average='weighted')
    f1_score_svm = f1_score(gasLabel[test_index], predict_svm, average='weighted')
    accuracy_svm = accuracy_score(gasLabel[test_index], predict_svm)
    print(rs_svm, precision_svm, f1_score_svm, accuracy_svm)
    rss_svm.append(rs_svm); rss_rf.append(rs_rf)
mean_rs_svm = np.mean(rss_svm); mean_rs_rf = np.mean(rss_rf)
print('mean SVM recall score: {:.5f}, mean RF recall score: {:.5f}'.format(mean_rs_svm, mean_rs_rf))
