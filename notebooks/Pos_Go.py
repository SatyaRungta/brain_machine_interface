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

fname = 'type/Nrn_cell_pos.mat'
fpath = root_path+folder_path+fname
data_dict2 = load_scipymat(fpath)

# Extracting variables from dictionary
# From dataset1
SPtimeTarget = data_dict1['SPtimeTarget']
SPtimeGo = data_dict1['SPtimeGo']
SPtimeMo = data_dict1['SPtimeMo']
# From dataset2
stime = np.transpose(data_dict2['time'])
Activity = data_dict2['Activity']
POS = data_dict2['POS']
RT  = data_dict2['RT']
HT  = data_dict2['HT']

N = Activity.shape[0] # No. of neurons
eventtarget, eventgo, eventmo = [0, 1, 2]    # Events

# Rasters and spike density functions
event         = eventgo
spike_timings = SPtimeGo
all_pos       = 8
sub = [4,1,2,3,6,9,8,7]

for neuron_id in range(0,10,1):

    goLoc = POS[:,neuron_id]
    goLoc = goLoc[~np.isnan(goLoc)]
    raster = spike_timings[neuron_id]

    posActivity = Activity[neuron_id,event]
    plt.subplots(3, 3, figsize=(10,8),layout='tight',sharex=True, sharey=True)

    for pos in range(0,all_pos,1):

        sdf  = posActivity[:,pos]
        trials = np.where(goLoc==pos)[0]
        posRaster = raster[trials,:]
        rt  = RT[trials,neuron_id]
        ht  = HT[trials,neuron_id]

        plt.subplot(3,3,sub[pos])
        plt.plot(posRaster,np.cumsum(np.ones(posRaster.shape),axis=0),marker='|',linestyle='none',color=[.8, .8, .8],markersize=1)
        plt.plot(rt, np.cumsum(np.ones(rt.shape), axis=0), marker='o', linestyle='none',color='g', markersize=1.5)
        plt.plot(-ht, np.cumsum(np.ones(ht.shape), axis=0), marker='^', linestyle='none', color='b', markersize=1.5)
        plt.plot(stime,sdf,color='k',linewidth=1)
        plt.fill_between([-250, 50], [250, 250], 0, facecolor=[.8, .8, .8], alpha=0.2)
        plt.vlines(0,0,225,colors='k',linestyles='--')
        plt.xlim(-1250,750)
        plt.ylim(0,225)
        plt.xticks(ticks=list(range(-1250,500,250)),labels=list(range(-1250,500,250)))
        plt.xlabel('Time from go (in ms)')
        plt.ylabel('Activity (sp/s)')

    plt.suptitle('Activity at different target locations for neuron '+str(neuron_id+1))

    figfolder = 'results/posActivity/'
    figname = 'Neuron' + str(neuron_id + 1) + '.jpg'
    plt.savefig(root_path + figfolder + figname)

    plt.close()
