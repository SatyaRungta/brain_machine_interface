"""
RT and HT distributions for the given sessions
"""
import numpy as np
from mat4py import loadmat
from tools.load_matfiles import *
import os

# Figure for histograms
import matplotlib.pyplot as plt
from visualise.plothist import *

# Required files
# Loading Dataset
root_path    = 'D:/projects/python/ADS/imi/brain_machine_interface/'
folder_path  = 'data/'

fname        = 'FeF1.mat'
fpath = root_path+folder_path+fname
data_dict1 = load_matfile(fpath)

fname = 'type/Nrn_cell_mf.mat'
fpath = root_path+folder_path+fname
data_dict2 = load_scipymat(fpath)

# Extracting variables from dictionary
# From dataset1
SPtimeTarget = data_dict1['SPtimeTarget']
SPtimeGo = data_dict1['SPtimeGo']
SPtimeMo = data_dict1['SPtimeMo']
# From dataset2
stime = np.transpose(data_dict2['time'])
Mf_activity = data_dict2['Mf_activity']

N = Mf_activity.shape[0] # No. of neurons
eventtarget, eventgo, eventmo = [0, 1, 2]    # Events

# Rasters and spike density functions
event = eventmo
spike_timings = SPtimeMo
iRF, oRF = [0, 1]
for neuron_id in range(0,N,1):
    raster = spike_timings[neuron_id]
    Mf    = Mf_activity[neuron_id,event]
    inMf  = Mf[:,iRF]
    outMf = Mf[:,oRF]

    plt.figure()
    plt.plot(raster,np.cumsum(np.ones(raster.shape),axis=0),marker='|',linestyle='none',color=[.8, .8, .8],markersize=1)
    plt.plot(stime,inMf,color='g')
    plt.plot(stime,outMf,color='r')
    plt.plot([0,0],[0,250],color='k')
    plt.xlim(-1500,500)
    plt.ylim(0,225)
    plt.xlabel('Time from go (in ms)')
    plt.ylabel('Activity (sp/s)')
    plt.title('Neuron id:'+str(neuron_id+1))
    plt.show()
