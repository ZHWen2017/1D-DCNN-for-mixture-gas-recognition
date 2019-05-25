import numpy as np
import pandas as pd
from ESN_handle import ESN

def correctConcentrations(conc_ideal):
    '''
    Returns the actual concentrations given ideal concentrations by accounting
    for delays and time flows.
    '''
    conc_real = np.zeros(conc_ideal.shape)
    # Delay in flow
    flow = 100 # cm3/min
    diameter = (3./16)*2.54 # cm
    length = 2.5*100 # cm
    volume = length*np.pi*(diameter**2/4.)
    delay = volume/flow*60 # seconds
    print("Delay: {0}sec".format(delay))
    freq = 100 # Hz
    t_step_offset = np.round(delay*freq)
    # print('t_step_offset: ', t_step_offset)
    #CO/Methane have a higher flow rate 200 cm3/min
    conc_real[int(np.round(t_step_offset/1.5)):,0] = conc_ideal[:-int(np.round(t_step_offset/1.5)),0]
    #Ethylene has a lower flow rate 100 cm3/min
    conc_real[int(t_step_offset):,1] = conc_ideal[:-int(t_step_offset),1] # Ethylene
    #conc_real = conc_ideal
    return conc_real

def add_features(data):
    data_temp = (data[:, :8] + data[:, 8:]) / 2
    features = np.zeros((data_temp.shape[0], data_temp.shape[1]//2))
    for i in range(features.shape[1]):
        features[:, i] = (data_temp[:, i*2] + data_temp[:, i*2+1]) / 2
    return features

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

sensor_data0 = sensor_data0[1000::timescale]
odors_data0 = odors_data0[1000::timescale]
times0 = times0[1000::timescale]
sensor_data_CO = sensor_data1[1000::timescale]
odors_data_CO = odors_data1[1000::timescale]
times1 = times1[1000::timescale]
# sensor_data_CO = np.hstack((sensor_data_CO,add_features(sensor_data_CO)))
# sensor_data_CO = add_features(sensor_data_CO)
print(sensor_data_CO.shape)




# print(sensor_data0.shape, odors_data0.shape, times0.shape,
#       sensor_data_CO.shape, odors_data_CO.shape, times1.shape)

# odors_data_CO = correctConcentrations(odors_data_CO)

esn = ESN(n_inputs=16, n_outputs=2,
          n_reservoir=600, spectral_radius=0.01, random_state=42, silent=False)

train_len = int(len(odors_data_CO) * 0.8)
test_index = 10000
esn.fit(sensor_data_CO[:train_len], odors_data_CO[:train_len])
prediction = esn.predict(sensor_data_CO[train_len:])
print("CO test RMSE: \n"+str(np.sqrt(np.mean(np.square((prediction[:, 0] - odors_data_CO[train_len:][:, 0]))))))
print("Eth test RMSE: \n"+str(np.sqrt(np.mean(np.square((prediction[:, 1] - odors_data_CO[train_len:][:, 1]))))))
# print(prediction.shape)
