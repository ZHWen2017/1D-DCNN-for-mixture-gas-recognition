import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import label_binarize
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
length = 100
# ##The data preprocessing
# f1 = open("../train_data_1.txt", "rb")
# gas_train = pickle.load(f1)
# # gas_train['label'] = np.array(gas_train['label'])
# f1.close()
#
# f2 = open("../test_data_1.txt", "rb")
# gas_test = pickle.load(f2)
# # gas_test['label'] = np.array(gas_test['label'])
# f2.close()
# gas_train['label'] = np.delete(gas_train['label'], 211, 0)
# gas_train['gas'] = np.delete(gas_train['gas'], 211, 0)

# f = open("C:/Users/Administrator/Desktop/resultPlot/dataPlot/gas_100.txt", "rb")
f = open("/home/wzh/PycharmProjects/data/multilabel/dataPlot/gas_100.txt", "rb")
gas = pickle.load(f)
# gas_test['label'] = np.array(gas_test['label'])
print(np.shape(gas['label']), np.shape(gas['gas']))
f.close()

Gas = {}
Gas['label'] = gas['label']
Gas['data'] = gas['gas']

# iterStep = 10
# valid_score_linear=[]; valid_score_rbf=[]; valid_score_poly=[]; valid_score_rnd=[]
# for step in range(iterStep):
##data Normalization
for i in range(len(Gas['data'])):
    gas_temp1 = Gas['data'][i]
    gas_mean1 = np.mean(gas_temp1, axis=0)
    # print(gas_mean1)
    gas_std = np.std(gas_temp1, axis=0)
    for j in range(len(gas_mean1)):
        gas_temp1[:, j] = (gas_temp1[:, j] - gas_mean1[j]) / gas_std[j]
    # gas_train={}; gas_test={}
    # gas_train['gas'], gas_test['gas'], gas_train['label'], gas_test['label'] \
    #     = train_test_split(gas['sensor_data'], gas['labels'], test_size=0.2)
    # print(gas_train['label'].shape, gas_train['gas'].shape,
    #       gas_test['gas'].shape, gas_test['label'].shape)

    # for i in range(len(gas_train['gas'])):
    #     gas_temp = gas_train['gas'][i]
    #     mean_value = np.mean(gas_temp, axis=0)
    #     std_value = np.std(gas_temp, axis=0)
    #     # print(mean_value.shape, std_value.shape)
    #     for j in range(len(mean_value)):
    #         gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]
    # # print(gas_train['gas'][3].shape)
    #
    #    ###The normalization of test dataset
    # for i in range(len(gas_test['gas'])):
    #     gas_temp1 = gas_test['gas'][i]
    #     gas_mean1 = np.mean(gas_temp1, axis=0)
    #     # print(gas_mean1)
    #     gas_std = np.std(gas_temp1, axis=0)
    #     for j in range(len(gas_mean1)):
    #         gas_temp1[:, j] = (gas_temp1[:, j] - gas_mean1[j]) / gas_std[j]
#
# def reSetLabel(gas_train):
#     n0, m0 = np.shape(gas_train['label'])
#     gas_train['label_co'] = []
#     for i in range(n0):
#         if (gas_train['label'][i][0] == 1 and gas_train['label'][i][1] == 0
#                 and gas_train['label'][i][2] == 0):
#             gas_train['label_co'].append(0)
#             # print('append 1: ', i)
#         elif (gas_train['label'][i][0] == 0 and gas_train['label'][i][1] == 1
#                 and gas_train['label'][i][2] == 0):
#             gas_train['label_co'].append(1)
#             # print('append 2: ', i)
#         elif (gas_train['label'][i][0] == 0 and gas_train['label'][i][1] == 0
#                 and gas_train['label'][i][2] == 1):
#             gas_train['label_co'].append (2)
#             # print('append 3: ', i)
#         elif (gas_train['label'][i][0] == 1 and gas_train['label'][i][1] == 1
#                 and gas_train['label'][i][2] == 0):
#             gas_train['label_co'].append(3)
#             # print('append 4: ', i)
#         elif (gas_train['label'][i][0] == 1 and gas_train['label'][i][1] == 0
#                 and gas_train['label'][i][2] == 1):
#             gas_train['label_co'].append(4)
#             # print('append 5: ', i)
#         else:
#             print('append emty')
#     return gas_train['label_co']

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
gasDataRe = np.reshape(np.array(Gas['data']),[593,-1])
pca = PCA(n_components=300)
pca.fit(gasDataRe)
gasDataRe = pca.transform(gasDataRe)
gasLabel = Gas['label_co']

model = OneVsOneClassifier(svm.SVC(kernel='linear', gamma=0.1, C=1))
    # clf = model.fit(gasDataReTrain, gas_train['label_co'])
linear_score = cross_val_score(model, gasDataRe, gasLabel, cv=10)
linear_score_mean = np.mean(linear_score)
print('\nAfter 10 hold cross validation, linear kernel test score: %.7f' % (linear_score_mean))
    # score_linear = clf.score(gasDataReTest, gas_test['label_co'])
    # print('linear kernel train score: %.6f' % clf.score(gasDataReTrain, gas_train['label_co']))
    # print('step# %d, linear kernel test score: %.7f' % (step, score_linear))

model1 = OneVsOneClassifier(svm.SVC(kernel='rbf', gamma=0.01, C=1, degree=4))
    # clf1 = model1.fit(gasDataReTrain, gas_train['label_co'])
    # score_rbf = clf1.score(gasDataReTest, gas_test['label_co'])
rbf_score = cross_val_score(model1, gasDataRe, gasLabel, cv=10)
rbf_score_mean = np.mean(rbf_score)
print('After 10 hold cross validation, RBF kernel test score: %.7f' % (rbf_score_mean))

model2 = OneVsOneClassifier(svm.SVC(kernel='poly', degree=3, coef0=1, C=5))
poly_score = cross_val_score(model2, gasDataRe, gasLabel, cv=10)
poly_score_mean = np.mean(poly_score)
print('After 10 hold cross validation, poly kernel test score: %.7f' % (poly_score_mean))

model_rnd = RandomForestClassifier(n_estimators=500)
rnd_score = cross_val_score(model_rnd, gasDataRe, gasLabel, cv=10)
rnd_score_mean = np.mean(rnd_score)
print('After 10 hold cross validation, RF test score: %.7f' % (rnd_score_mean))

#  print('rbf kernel train score: %.6f' % clf1.score(gasDataReTrain, gas_train['label_co']))
#     print('step# %d, rbf kernel test score: %.7f' % (step, score_rbf))
#
#     model1 = OneVsOneClassifier(svm.SVC(kernel='poly', degree=3, coef0=2, C=1))
#     clf1 = model1.fit(gasDataReTrain, gas_train['label_co'])
#     score_poly = clf1.score(gasDataReTest, gas_test['label_co'])
#     # print('rbf kernel train score: %.6f' % clf1.score(gasDataReTrain, gas_train['label_co']))
#     print('step# %d, poly kernel test score: %.7f' % (step, score_poly))
#
#     model_rnd = RandomForestClassifier(n_estimators=400)
#     clf_rnd = model_rnd.fit(gasDataReTrain, gas_train['label_co'])
#     score_rnd = model_rnd.score(gasDataReTest, gas_test['label_co'])
#     print('step# %d, random forest test score: %.7f' % (step, score_rnd))
#
#     valid_score_linear.append(score_linear)
#     valid_score_rbf.append(score_rbf)
#     valid_score_poly.append(score_poly)
#     valid_score_rnd.append(score_rnd)
# valid_score_linear_mean = np.mean(valid_score_linear)
# valid_score_rbf_mean = np.mean(valid_score_rbf)
# valid_score_poly_mean = np.mean(valid_score_poly)
# valid_score_rnd_mean = np.mean(valid_score_rnd)
# print('\nAfter %d iter: ' % iterStep)
# form = [valid_score_linear_mean, valid_score_rbf_mean, valid_score_poly_mean, valid_score_rnd_mean]
# print('linear mean valid score={:.6f}, rbf mean valid score={:.6f}, poly mean valid score={:.6f}, rnd mean valid score={:.6f}' .format(*form))
