import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set(style="ticks", palette="muted")
matplotlib.rcParams['xtick.direction'] = 'in'   #set the direction of the ticks
matplotlib.rcParams['ytick.direction'] = 'in'

# test_result = 'C:/Users/Administrator/Desktop/resultPlot/BNModel/result/test_result_cpu_jour.txt'
test_result = './test_result_conference.txt'
f1 = open(test_result, 'rb')
result = pickle.load(f1)
f1.close()


trainLoss = result['train_loss'];testLoss = result['test_loss']
testAcc = result['acc']; acc_best = result['acc_best']
trainAcc = result['train_acc']

best_2000 = 0
for i in range(len(testAcc[:2000])):
    if testAcc[i] > best_2000:
        best_2000=testAcc[i]
print('2000 iteration best acc: ', best_2000)
print('5000 iteration best acc:', acc_best)
###画图
length = len(testAcc[:2000])
X = np.arange(length)
fig = plt.figure(figsize=(10,8), dpi=80)
# fig.patch.set_alpha() #设置背景透明度
axes = plt.subplot(111)


font = {'family': 'serif',
        'weight': 'bold',
        'size': 24,
        }
###绘制accuracy图
x_tick = np.arange(0, length+200, 200)
y_tick = np.arange(0, 1.1, 0.1)

matplotlib.rc('font', **font)

axes.plot(X[::15], testAcc[:2000][::15], 'b-', label='test', linewidth=1.2, )
axes.plot(X[::15], trainAcc[:2000][::15], 'c-', label='train', linewidth=1.2, )
axes.set_xticks(x_tick);axes.set_yticks(y_tick)
axes.set_xlabel('Iterations', fontdict=font)
axes.set_ylabel('Accuracy',fontdict=font)
# axes.set_title('Gas Accuracy', fontdict=font, fontsize=25)
axes.yaxis.set_ticks_position('both')
axes.xaxis.get_label()
# axes.yaxis.set_ticks_position('left')
for tick in axes.xaxis.get_major_ticks():
    tick.label1On = True
    tick.label2On = False
    tick.label1.set_fontsize(15)
for tick in axes.yaxis.get_major_ticks():
    tick.label1On = True
    tick.label2On = False
    tick.label1.set_fontsize(15)


legend=axes.legend(loc = 'lower right', shadow=True, fontsize='medium', frameon=True,
                   edgecolor='k')

result_image = './resultPlot'
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
plt.savefig(result_image+'/Accuracy_100_conf.svg')
plt.show()


##绘制Loss图
fig1 = plt.figure(figsize=(10,8), dpi=80)
axes1 = fig1.add_subplot(111)
matplotlib.rc('font', **font)

axes1.plot(X[::12], testLoss[:2000][::12], 'b-', label='test', linewidth=1.5,)
axes1.plot(X[::12], trainLoss[:2000][::12], 'c-', label='train', linewidth=1.5,)
axes1.set_xlabel('Iterations', fontdict=font)
axes1.set_ylabel('Loss',fontdict=font)
# axes.set_title('Gas Accuracy', fontdict=font, fontsize=25)
axes1.yaxis.set_ticks_position('both')

for tick in axes1.xaxis.get_major_ticks():
    tick.label1On = True
    tick.label2On = False
    tick.label1.set_fontsize(15)
for tick in axes1.yaxis.get_major_ticks():
    tick.label1On = True
    tick.label2On = False
    tick.label1.set_fontsize(15)


axes1.legend(loc = 'best', shadow=True, fontsize='medium', frameon=True,
                   edgecolor='k')
result_image = './resultPlot'
plt.savefig(result_image+'/Loss_100_conf.svg')
plt.show()
plt.close()
