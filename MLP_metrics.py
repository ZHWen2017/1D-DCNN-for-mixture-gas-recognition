import numpy as np
# from MLPwithBN import *
import pickle
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools
import os
Max_steps = 2000

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# gas_path = 'C:/Users/Administrator/Desktop/resultPlot/dataPlot/gas_100.txt'
gas_path = '/home/wzh/PycharmProjects/data/multilabel/dataPlot/gas_100.txt'
###绘制混淆矩阵图
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize='large')
    plt.xlabel('Predicted label',fontsize='large')
    save_path = 'C:/Users/Administrator/Desktop/对比算法/result_fig/'
    plt.savefig(title+'.eps')
    plt.show()

def ConcatData(gasdata):
    aX0, aX1, aX2 = np.shape(gasdata.transpose(0, 2, 1))
    gasDataRe = []
    gasdata = gasdata.transpose(0, 2, 1)
    for i in range(aX0):
        gasMartix = gasdata[i]
        gas_temp = gasMartix[:, 0][::10]
        for j in range(aX2 - 1):
            gas_temp = np.concatenate((gas_temp, gasMartix[:, j+1][::10]))
        gasDataRe.append(gas_temp)
    return gasDataRe

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

f1 = open(gas_path, 'rb')
gas_data = pickle.load(f1)
f1.close()
print(gas_data['gas'].shape, np.shape(gas_data['label']))
for i in range(len(gas_data['gas'])):
    gas_temp = gas_data['gas'][i]
    mean_value = np.mean(gas_temp, axis=0)
    std_value = np.std(gas_temp, axis=0)
    # print(mean_value.shape, std_value.shape)
    for j in range(len(mean_value)):
        gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]
# print(gas_train['gas'][3].shape)



gas_data['label_'] = reSetLabel(gas_data)
gas_data['gas'] = np.reshape(np.array(gas_data['gas']), [593, -1])
pca = PCA(n_components=300)
pca.fit(gas_data['gas'])
gasData = np.array(pca.transform(gas_data['gas']))
gasLabel = np.array(gas_data['label_'])

# X_train, X_test, y_train, y_test = train_test_split(gasData, gasLabel, test_size=0.2, random_state=10)
# labels = list(set(y_test))
# print(y_test)
# feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
# dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=6,
#                                          feature_columns=feature_columns)
# dnn_clf.fit(x=X_train, y=y_train, batch_size=64, steps=2000)
# y_pred = list(dnn_clf.predict(X_test))
# class_report = classification_report(y_test, y_pred)
# conf_mat = confusion_matrix(y_test, y_pred)
# target_names = ['Air','Eth','CO', 'Met','Eth-CO','Eth-Me']
# test_result = {}
# test_result['ANN']=np.array([y_test,y_pred])
# test_path='C:/Users/Administrator/Desktop/对比算法/result_fig/test_result_ANN.txt'
# f1 = open(test_path, 'wb')
# pickle.dump(test_result, f1)
# f1.close()
# print(class_report)
# plot_confusion_matrix(conf_mat, classes=target_names, normalize=True, title='ANN Normalized confusion matrix')

print('       10 hold Cross validating .....')
test_accs=[]
KF = KFold(n_splits=10,random_state=2018)
for train_index, test_index in KF.split(X=gasData):
    train_set = {'gas':[], 'label':[]}; test_set = {'gas':[], 'label':[]}
    for index0 in train_index:
        train_set['gas'].append(gasData[index0])
        train_set['label'].append(gasLabel[index0])
    for index1 in test_index:
        test_set['gas'].append(gasData[index1])
        test_set['label'].append(gasLabel[index1])
    train_set['gas'] = np.array(train_set['gas']); train_set['label'] = np.array(train_set['label'])
    test_set['gas'] = np.array(test_set['gas']) ; test_set['label'] = np.array(test_set['label'])

    print(np.shape(train_set['gas']), np.shape(train_set['label']),
          np.shape(test_set['gas']), np.shape(test_set['label']))
    _, length = np.shape(train_set['gas'])
    # n_classes = np.unique(train_set['label'])
    ###build MLP classifier
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(train_set['gas'])
    dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 300], n_classes=6,
                                             feature_columns=feature_columns)
    dnn_clf.fit(x=train_set['gas'], y=train_set['label'], batch_size=64, steps=3000)

    y_pred = list(dnn_clf.predict(test_set['gas']))
    score = accuracy_score(test_set['label'], y_pred)
    test_accs.append(score)


    # test_acc, best_acc = MLP(hold_i, train_set, test_set, Max_steps, length)
    # test_accs.append(best_acc)
    # hold_i+=1
    # print('the best accuracy = %.6f' % best_acc)
acc_mean = np.mean(test_accs)
print('\n\nAfter 10 fold Cross validation , the MLP mean accuracy: %.6f'% acc_mean)
