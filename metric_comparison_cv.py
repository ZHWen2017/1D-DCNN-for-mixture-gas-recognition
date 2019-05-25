import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import itertools
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

####绘制混淆矩阵图
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    save_path = 'C:/Users/Administrator/Desktop/对比算法/result_fig/'
    plt.savefig(title+'.eps')
    plt.show()

gasDataRe = np.reshape(np.array(Gas['data']),[582,-1])
pca = PCA(n_components=20)
pca.fit(gasDataRe)
gasDataRe = np.array(pca.transform(gasDataRe))
gasLabel = np.array(Gas['label_co'])
# print(list(set(gasLabel)))
# X_train, X_test, y_train, y_test = train_test_split(gasDataRe, gasLabel, test_size=0.2,random_state=10)
KFolds = KFold(n_splits=5, random_state=10)
# labels = list(set(y_test))
clfs = [svm.SVC(kernel='linear',gamma=0.1, C=1),
        RandomForestClassifier(n_estimators=200,n_jobs=-1, criterion='gini'),
        KNeighborsClassifier()]

classifier_name = ['SVM-linear','RandomForest', 'KNN']
for i, clf in enumerate(clfs):
    print(classifier_name[i]+' training')
    for train_index, test_index in KFolds.split(gasDataRe,gasLabel):
        X_train, y_train = gasDataRe[train_index], gasLabel[train_index]
        X_test, y_test = gasDataRe[test_index], gasLabel[test_index]
        labels = list(set(y_test))
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        y_score = clf.score(X_test, y_test)
        print(y_score)
        # print(list(set(y_predict)))
        con_mat = confusion_matrix(y_test, y_predict, labels=labels)
        target_names = ['Air','Eth','CO', 'Met','Eth-CO','Eth-Me']
        class_report = classification_report(y_test, y_predict,target_names=target_names)
        print(con_mat)
        # print(class_report.format_map('f1-score'))
        # plot_confusion_matrix(con_mat, classes=target_names, normalize=True, title='%s Normalized confusion matrix'%classifier_name[i])
# ####ROC curve
# y_train = label_binarize(y_train,classes=[0,1,2,3,4,5])
# y_test = label_binarize(y_test,classes=[0,1,2,3,4,5])
# classifiers = [OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True)),
#                OneVsRestClassifier(KNeighborsClassifier()),
#                OneVsRestClassifier(RandomForestClassifier(n_estimators=500, n_jobs=-1))]
# for classifier in classifiers:
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     classifier.fit(X_train,y_train)
#     y_score = classifier.predict_proba(X_test)
#     y_predict=classifier.score(X_test,y_test)
#     print(y_predict)
#     for i in range(len(labels)):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#     # print(roc_auc["micro"])
#     plt.figure()
#     lw = 2
#     plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()