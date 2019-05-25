import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

sns.set(style="ticks", palette="muted")
matplotlib.rcParams['xtick.direction'] = 'in'   #set the direction of the ticks
matplotlib.rcParams['ytick.direction'] = 'in'

# gas_path = 'C:/Users/Administrator/Desktop/resultPlot/dataPlot/gasraw.txt'
f = open('gasraw.txt', 'rb')
Gas = pickle.load(f)
f.close()
print(np.shape(Gas['gas'][11]))

index = 300
sensor_i = 11

gasPlot = Gas['gas'][index]
labelPlot = Gas['label'][index][0]
times = Gas['times'][index]
gas_dict = {'Air':[0, 0, 0], 'Eth':[1, 0, 0], 'CO':[0, 1, 0], 'Met':[0, 0, 1],
            'Eth-CO':[1, 1, 0], 'Eth-Met':[1, 0, 1]}
labelname=' '
for keyword, value in gas_dict.items():
    # print(keyword)
    if value == labelPlot:
        labelname=keyword


minlength = 100
length = len(gasPlot)
index_sort = np.arange((length-minlength))
gas_bigest = 0; gasBigSet = []; timeBigSet = []
for index_1 in index_sort:
    index_end = index_1 + minlength
    gas_def1 = np.absolute(gasPlot[index_1][sensor_i] - gasPlot[index_end][sensor_i])
    gas_dif1 = (np.divide(gas_def1, gasPlot[index_1][sensor_i]))
    if gas_dif1 > gas_bigest:
        gas_bigest = gas_dif1
for index_2 in index_sort:
    index_end = index_2 + minlength
    gas_def1 = np.absolute(gasPlot[index_2][sensor_i] - gasPlot[index_end][sensor_i])
    gas_dif1 = (np.divide(gas_def1, gasPlot[index_2][sensor_i]))
    if gas_dif1 == gas_bigest:
        gasBigSet=gasPlot[index_2:index_end][:, sensor_i]
        timeBigSet=times[index_2:index_end]
        print(np.shape(gasBigSet), np.shape(timeBigSet))

###绘制数据图， 10秒的最大梯度图
X = times
X1 = timeBigSet
fig = plt.figure(figsize=(10,8), dpi=80)
# fig.patch.set_alpha() #设置背景透明度
axes = plt.subplot(111)


font = {'family': 'serif',
        'weight': 'bold',
        'size': 22,
        }
###绘制accuracy图

matplotlib.rc('font', **font)
x0=X1[0];y0=gasBigSet[0];x1=X1[-1];y1=gasBigSet[-1]
axes.plot(X[::10], gasPlot[:, sensor_i][::10], 'k--', label=labelname, linewidth=1.5)
axes.plot(X1, gasBigSet, 'b-', label='Largest gradient', linewidth=1.5)
axes.scatter([x0,],[y0,], s=50, color='r')
axes.annotate(r'$[%.2f,%.2f]$'%(x0, y0), xy=(x0+2, y0-1),xycoords='data', fontsize=15,color='k')
axes.scatter([x1,],[y1,], s=50, color='r')
axes.annotate(r'$[%.2f,%.2f]$'%(x1, y1), xy=(x1+2, y1-1),xycoords='data', fontsize=15,color='k')
# axes.set_xticks(x_tick);axes.set_yticks(y_tick)
axes.set_xlabel('Times(S)', fontdict=font)
axes.set_ylabel('Conductivities',fontdict=font)
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


legend=axes.legend(loc = 'best', fontsize='small', frameon=True,
                   edgecolor='k')
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
image_path='C:/Users/Administrator/Desktop/resultPlot/result/grad_10s.svg'
plt.savefig(image_path)
plt.show()
