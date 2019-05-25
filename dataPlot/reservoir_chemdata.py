#Authors: Sadique Sheik / Jordi Fonollosa
#Biocircuits Institute; University of California San Diego
#email: {ssheik,fonollosa}@ucsd.edu 
# 
#This code is made free upon citation request of: 
#Reservoir computing algorithms compensate for the slow response of chemosensor arrays 
#to fast varying gas concentrations in continous monitoring;
#J Fonollosa, S Sheik, R Huerta, S Marco; Sensors and Actuators B; 2015

#The code runs Reservoir Computing algorithms on three different datasets. 
#Uncomment one of the lines in the code to select the dataset.
#dataset, mixture = 'hardware', 'ethy_meth' #Real data, mixture of Ethylene with Methane
#dataset, mixture = 'hardware', 'ethy_CO' #Real data, mixture of Ethylene with CO
#dataset, mixture = 'simulation', 'meth_CO' #Simulated data, mixture of Methane with CO
# 
#The datasets should be located in the folder data/ and have the following filenames:
#data/ethylene_methane.txt
#data/ethylene_CO.txt
#data/exper22.mat
#
#The first dataset is a dataset acquired from real MOX sensors exposed to mixtures of Ethylene and Methane in air
#The second dataset is a dataset acquired from real MOX sensors exposed to mixtures of Ethylene and CO in air
#The third dataset is a dataset simulated from MOX sensors exposed to mixtures of CO and Methane
#
#Classification accuracy is provided in Results/_Error
#
#Better description on the datasets and algorithm can be found in the above mentioned paper.



  




import os
# import Oger
# import mdp
import numpy as np
import scipy as sp
from scipy.io import loadmat
import random
import pylab
import sys
# from lvnode import LVReservoirNode, MyLRReadoutNode, MyPerceptronReadoutNode

import matplotlib
matplotlib.rcParams.update({'font.size': 13})


def loadRawData(set="hardware", timescale=10, dataset=16, cross=False):
    '''
    This function should load all data from original data files and return it.
    set: simulation/hardware
    '''
    if set=='simulation':
        x = loadmat('data/exper22.mat')
        sensory_outs = x['RRq']
        c1 = x['c1']
        c2 = x['c2']
        odors = np.append(c1,c2,axis=1)
    elif set=='hardware':
        if dataset==11:
            x = np.loadtxt('../../ethylene_methane.txt', skiprows=1)
        else:
            x = np.loadtxt('../../ethylene_CO.txt', skiprows=1)
        #x = np.load('hwdata/june'+str(dataset)+'_test4.npy')
        # odors = x
        print(np.shape(x))
        sensory_outs = x[:,3:]
        odors = x[:,1:3]
        # Normalize concentrations
        odors[:,0] = odors[:,0] # 16 -> 50 , 11 ->10
        #return (sensory_outs[1000::10],), odors[1000::10]
        # odors = correctConcentrations(odors)

    # Discard initial readings where the sensor initializes
    s = sensory_outs[1000::timescale]
    o = odors[1000::timescale]

    if cross:
        rollindx = np.random.randint(len(s))
    else:
        rollindx = 0
    return (np.roll(s, rollindx, axis=0),), np.roll(o, rollindx, axis=0)

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
    #CO/Methane have a higher flow rate 200 cm3/min
    conc_real[int(np.round(t_step_offset/1.5)):,0] = conc_ideal[:-int(np.round(t_step_offset/1.5)),0]
    #Ethylene has a lower flow rate 100 cm3/min
    conc_real[t_step_offset:,1] = conc_ideal[:-t_step_offset,1] # Ethylene
    #conc_real = conc_ideal
    return conc_real

def loadData(set='hardware', hwset=11, cset=0):
    '''
    set : hardware/simulation
    hwset: 11/16
    cset : 0 .. 19
    if cset = -1, return original data
    '''
    loc = 'data/'
    if cset == -1:
        return loadRawData(set=set,dataset=hwset,cross=False)
    # if set=='hardware':
    #     x = loadmat(loc+set+'/'+str(hwset)+'/'+str(cset)+'.mat')
    # else:
    #     x = loadmat(loc+set+'/'+str(cset)+'.mat')
    #
    # return (x['sensordata'],), x['odordata']


def computeError(out,data, onsets=False, perc=False):
    '''
    Return the computed standard error
    '''
    indx = 0
    if onsets == True:
        err = []
        while True:
            try:
                indx0 = np.nonzero(data[indx:,0] != data[indx,0])[0][0]
                indx1 = np.nonzero(data[indx:,1] != data[indx,1])[0][0]
                #indx0 = np.nonzero(
                #    np.logical_and(
                #        (data[indx:,0]!=data[indx,0]),
                #        (data[indx:,0] != 0)))[0][0]
                #indx1 = np.nonzero(
                #    np.logical_and(
                #        (data[indx:,1] != data[indx,1]),
                #        (data[indx:,1] != 0)))[0][0]
                indx += min(indx0, indx1)
                #print indx0, indx1, indx
                err.append(computeError(out[indx:indx+30], 
                                    data[indx:indx+30]))
            except IndexError as e:
                break
        msq = np.sqrt(np.sum(np.square(err), axis=0)/len(err))
    else:
        msq = np.sqrt(np.sum(np.square(out-data), axis=0)/len(data))
    if perc:
        msq = msq/data.max(axis=0)*100.
    return msq


def computeAllErrors(odordata, prediction, error_offset=1000,
                     filepath='Results/Cross/Test', savedata=False):
    '''
    Compute errors and saves to a file
    error_offset: initial timesteps to exclude
    filepath: location of file to which data needs to be appended.
    '''
    err = computeError(prediction[error_offset:], 
                       odordata[error_offset:],
                       perc=False)
    err_p = computeError(prediction[error_offset:], 
                       odordata[error_offset:],
                       perc=True)
    on_err = computeError(prediction[error_offset:], 
                          odordata[error_offset:],
                          onsets=True,perc=False)
    on_err_p = computeError(prediction[error_offset:], 
                          odordata[error_offset:],
                          onsets=True,perc=True)
    
    if savedata:
        with open(filepath+'_Error', 'a') as f:
            np.savetxt(f, [err])
        with open(filepath+'_OnError', 'a') as f:
            np.savetxt(f, [on_err])
    print("Prediction error is {0}".format(err))
    print("Prediction error is {0} percent".format(err_p))
    print("Onset Prediction error is {0}".format(on_err))
    print("Onset Prediction error is {0} percent".format(on_err_p))


def segmentTrainTestData(sensordata, odordata, train_frac=0.6):
    '''
    trian_frac : fraction of data to be used for training
    '''
    input_dim = sensordata[0].shape[1]
    print(sensordata[0].shape)
    N_odor = odordata.shape[1] # 2 different odors
    
    # Input parameters
    train_frac = 0.6
    n_samples = sensordata[0].shape[0]
    print("samples is: ", n_samples)
    n_train_samples = int(round(n_samples*train_frac))
    n_test_samples = n_samples - n_train_samples
    
    zd = zip([sensordata[0][0:n_train_samples]],
                        [odordata[0:n_train_samples]])
    #zd *= 4 # Present the same data multiple times
    data = [zd]
    return data, input_dim, N_odor, n_samples, n_train_samples, n_test_samples


def runLinearClassifier(sensordata, odordata, test='test'):
    '''
    Run a linear classifier on the given data and return the predicted odor concentrations
    '''
    data, input_dim, N_odor, n_samples, n_train_samples, n_test_samples = segmentTrainTestData(sensordata, 
                                                                                               odordata, 
                                                                                               train_frac=0.6)
    # Define flow
    readout = MyLRReadoutNode(input_dim=input_dim,
                            output_dim=N_odor)

    flow = mdp.Flow([readout])
    
    # Training
    print('Training on data')
    flow.train(data)
    
    if test=='test':
        # Generate output for test data
        prediction = flow.execute(sensordata[0][n_train_samples:])
        return prediction, odordata[n_train_samples:]
    elif test=='train':
        prediction = flow.execute(sensordata[0][:n_train_samples])
        return prediction, odordata[:n_train_samples]

def plotReservoir(flow,data,odors,out,dataset='hardware',hwset=16):
    '''
    Plot the input, reservoir states and output of the flow for a given data
    data: data to be fed in the flow
    odors: plot actual odor concentration
    output: output of the flow
    '''
    TIMERESCALE = 10.
    # dataset = hwset
    dt = 1./100.*TIMERESCALE # 100 Hz
    t = np.arange(0,len(odors)*dt, dt) #- 4000
    out = out
    #pylab.figure()

    # Sensor output
    if dataset == 'simulation':
        f,ax = pylab.subplots(4,sharex=True)
        ax[0].plot(t, data/1000)
        ax[0].set_ylabel('$S_{data}$ \n $(K\Omega)$')
        ax[0].set_yticks([0,50,100])
        #ax[0].set_yticklabels(['0','3','6'])
        ax[1].plot(t, flow[0].inspect()[0][:,:10])
        ax[1].set_ylabel('Reservoir\n states')
        ax[1].set_yticks([-0.4,0,0.4])
        ax[2].plot(t, odors[:,0], 'b-')
        ax[2].plot(t, out[:,0], 'b-', alpha=0.3)
        #ax[2].set_yticks([0,150,300])
        ax[2].set_xlim((0,2400))
        ax[2].set_ylim((-10,80))
        ax[2].set_ylabel('Methane \n (ppm)',color='b')
        ax[2].set_yticks([0,30,60])
        ax[2].legend(['$CH_4$ setpoint', '$CH_4$ prediction'])
        ax[3].plot(t, odors[:,1],'g-')
        ax[3].plot(t, out[:,1],'g-',alpha=0.3)
        ax[3].set_xlabel('Time (s)')
        ax[3].set_ylabel('CO \n (ppm)',color='g')
        ax[3].set_yticks([0,30,60])
        ax[3].set_ylim((-10,80))
        ax[3].legend(['$CO$ setpoint', '$CO$ prediction'])
    else:
            
            
            
        if hwset == 11:
            f,ax = pylab.subplots(4,sharex=True)
            ax[0].plot(t-3700, data)
            ax[0].set_ylabel('$S_{data}$ \n $(\Omega^{-1}$ \ $\Omega_0^{-1})$',fontsize=18)
            ax[0].set_yticks([0,3000,6000])
            ax[0].set_yticklabels(['0','3','6'])
        else:
            f,ax = pylab.subplots(4,sharex=True)
            ax[0].plot(t-8300, data)
            ax[0].set_ylabel('$S_{data}$ \n $(\Omega^{-1}$ \ $\Omega_0^{-1})$',fontsize=18)
            ax[0].set_ylim(0,12000)
            ax[0].set_yticks([0,6000,12000])
            ax[0].set_yticklabels(['0','6','12'])
    
        #ax[0].set_xticks([0,2000, 4000,6000, 8000])
        
        # Reservoir states
        #axc = pylab.subplot(4,1,2, sharex=ax)
        if hwset == 11:
            ax[1].plot(t-3700, flow[0].inspect()[0][:,:10])
            ax[1].set_ylabel('Reservoir\n states',fontsize=18)
            ax[1].set_yticks([-0.4,0,0.4])
        else:
            ax[1].plot(t-8300, flow[0].inspect()[0][:,:10])
            ax[1].set_ylabel('Reservoir\n states',fontsize=18)
            ax[1].set_yticks([-0.4,0,0.6])
    
        #axc.set_xticks([])
    
    
        # Gas concentrations
    
        if hwset == 11:
            ax[2].plot(t-3700, odors[:,0], 'b-')
            ax[2].plot(t-3700, out[:,0], 'b-', alpha=0.3)
            ax[2].set_ylabel('Methane\n (ppm)',color='b',fontsize=18)
            ax[2].set_yticks([0,150,300])
            ax[2].set_xlim((0,8000))
            ax[2].set_ylim((-50,350))
            #ax[2].legend(['$CH_4$ setpoint', '$CH_4$ prediction'])
            ax[3].plot(t-3700, odors[:,1],'g-')
            ax[3].plot(t-3700, out[:,1],'g-',alpha=0.3)
            ax[3].set_xlabel('Time (s)',fontsize=18)
            ax[3].set_ylabel('Ethylene\n (ppm)',color='g',fontsize=18)
            ax[3].set_yticks([0,10,20])
            ax[3].set_ylim((-5,25))
            #ax[3].legend(['$C_2H_4$ setpoint', '$C_2H_4$ prediction'])
        else:
            ax[2].plot(t-8300, odors[:,0], 'b-')
            ax[2].plot(t-8300, out[:,0], 'b-', alpha=0.3)
            ax[2].set_ylabel('CO\n (ppm)',color='b',fontsize=18)
            ax[2].set_yticks([0,300, 600])
            ax[2].set_xlim((0,8000))
            ax[2].set_ylim((-100,650))
            #ax[2].legend(['$CO$ setpoint', '$CO$ prediction'])
            ax[3].plot(t-8300, odors[:,1],'g-')
            ax[3].plot(t-8300, out[:,1],'g-',alpha=0.3)
            ax[3].set_xlabel('Time (s)',fontsize=18)
            ax[3].set_ylabel('Ethylene\n (ppm)',color='g',fontsize=18)
            ax[3].set_yticks([0,10,20])
            ax[3].set_ylim((-5,25))
            #ax[3].legend(['$C_2H_4$ setpoint', '$C_2H_4$ prediction'])
        
    #ax1.set_xticks([])
    #pylab.legend(['Ethylene', 'Methane']) #11
    #pylab.legend(['Ethylene', 'CO']) #16
    #pylab.legend(['Methane', 'CO']) # Simulation
    
    # Output concentrations

    #ax[3].set_xlim((0,8000))
    
    #for tl in ax[3].get_yticklabels():
    #    tl.set_color('b')
    #ax[3].set_ylim(lim1)
    
    #ax2 = ax[3].twinx()
    #ax2.plot(t, out[:,1], 'g-')
    #ax2.set_ylabel('Est. Ethylene\n (ppm)',color='g')
    #for tl in ax2.get_yticklabels():
    #    tl.set_color('g')
    #ax2.set_ylim(lim2)
    #ax2.set_yticks([0,10,20])
    #for i in range(4):
    #    ax[i].grid(True)
    return out

    


def runReservoir(sensordata, odordata, input_scaling=1e-6, sr=0.9, output_dim = 100,test='test',
                plot=False,dataset='hardware',hwset=16):
    '''
    Run a reservoir on the given data and return the predicted odor concentrations
    simulation data : input_scaling=1e-6, sr=0.9
    hardware data : input_scaling=1e-5, sr=0.1
    test: test/train; test with test data set or trainig data set. Useful for cross validation training
    plot: plot results
    '''
    data, input_dim, N_odor, n_samples, n_train_samples, n_test_samples = segmentTrainTestData(sensordata, 
                                                                                               odordata, 
                                                                                               train_frac=0.6)
    data = [[],data[0]]
    
    reservoir = Oger.nodes.ReservoirNode(input_dim=input_dim,
                                    output_dim=output_dim,
                                    input_scaling=input_scaling, # HW: 1e-5
                                    spectral_radius=sr, 
                                   )
    readout = MyLRReadoutNode(input_dim=output_dim,
                            output_dim=N_odor)

    flow = reservoir + readout

    # Training
    print('Training on data')
    flow.train(data)

    if plot:
        Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)
    
    if test=='test':
        # Generate output for test data
        inputs = sensordata[0][n_train_samples:]
        prediction = flow.execute(inputs)
        results = prediction, odordata[n_train_samples:]
    elif test=='train':
        inputs = sensordata[0][:n_train_samples]
        prediction = flow.execute(sensordata[0][:n_train_samples])
        results = prediction, odordata[:n_train_samples]
    if plot:
        plotReservoir(flow, inputs, results[1], results[0], dataset=dataset,hwset=hwset)
    return results


def genTrainingCross():
    '''
    Train with cross validation data
    '''
    args = sys.argv[1:]
    print(args)
    if len(args) > 0:
        seed = int(args[0])
        np.random.seed(seed)
    else:
        pass   
    pylab.ion()
    dataset, hwset = 'hardware', 16
    if dataset == 'simulation':
        input_scaling=1e-6
        sr=0.9
    else:
        input_scaling=1e-5
        sr=0.1
    
    for cset in range(20):
        sensordata, odordata = loadData(set=dataset, hwset=hwset, cset=cset)
        #prediction, testodordata = runLinearClassifier(sensordata, odordata)
        #computeAllErrors(testodordata,prediction, 
        #                 filepath='Results/Cross/Linear/'+dataset+'_'+str(cset))
        prediction, testodordata = runReservoir(sensordata, odordata, 
                                                input_scaling=input_scaling,
                                                sr=sr,
                                                test='train')
        computeAllErrors(testodordata,prediction, 
                         filepath='Results/Cross/Reservoir/'+dataset+str(hwset)+'_'+str(cset),
                         savedata=True) #+str(hwset)+


def genTestCross():
    '''
    Run reservoir on test cross valiation data and compute errors.
    '''
    dataset, hwset = 'hardware', 11
    if dataset == 'simulation':
        input_scaling=1e-6
        sr=0.9
    else:
        input_scaling=1e-5
        sr=0.1
    for cset in range(20):
        # Load data
        sensordata, odordata = loadData(set=dataset, hwset=hwset, cset=cset)
        
        # Find seed for the minimum result
        trainerr = np.loadtxt('Results/Cross/Reservoir/'+dataset+str(hwset)+'_'+str(cset)+'_Error') #+str(hwset)+
        seed = trainerr.argmin(axis=0)[0] + 1 # Find seed for the minimum result
        np.random.seed(seed)

        
        # Run reservoir for results
        prediction, testodordata = runReservoir(sensordata, odordata, 
                                                input_scaling=input_scaling,
                                                sr=sr,
                                                test='test')
        computeAllErrors(testodordata,prediction, 
                         filepath='Results/Cross/Reservoir/test_'+dataset+str(hwset)+'_'+str(cset),
                         savedata=True) #+str(hwset)+



if __name__=='__main__':
    pylab.ion()
    #genTestCross()
    
    # SELECT DATASET: UNCOMMENT THE LINE THAT CORRESPONDS TO THE SELECTED DATASET:
    # dataset, mixture = 'hardware', 'ethy_meth' #Real data, mixture of Ethylene with Methane
    dataset, mixture = 'hardware', 'ethy_CO' #Real data, mixture of Ethylene with CO
    # dataset, mixture = 'simulation', 'meth_CO' #Simulated data, mixture of Methane with CO
    
    if mixture == 'ethy_CO':
        hwset=16
    else:
        hwset=11
    
    if dataset == 'simulation':
        input_scaling=1e-6
        sr=0.9
        output_dim=200
    else:
        input_scaling=1e-5
        sr=0.1
        output_dim=100
    sensordata, odordata = loadData(set=dataset, hwset=hwset, cset=-1)
    data, input_dim, N_odor, n_samples, n_train_samples, n_test_samples = segmentTrainTestData(sensordata,
                                                                                               odordata,
                                                                                               train_frac=0.6)
    print(input_dim, N_odor, n_samples, n_train_samples, n_test_samples)

    # Run reservoir for results
    prediction, testodordata = runReservoir(sensordata, odordata, 
                                            input_scaling=input_scaling,
                                            sr=sr,
                                            output_dim = output_dim,
                                            test='test', 
                                            plot=True,
                                            dataset=dataset,
                                            hwset=hwset)

    computeAllErrors(testodordata, prediction, error_offset=1000,
                     filepath='Results/', savedata=True)
