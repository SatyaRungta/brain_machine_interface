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
inHT, outHT = data_dict2['HT_MF']
inRT, outRT = data_dict2['RT_MF']
inPOS, outPOS = np.transpose(data_dict2['SPOS'])
MF  = data_dict2['MF']

N = Mf_activity.shape[0]                     # No. of neurons
eventtarget, eventgo, eventmo = [0, 1, 2]    # Events

# Raster and spike density functions
event = eventtarget
spike_timings = SPtimeTarget
iRF, oRF = [0, 1]
inHT, outHT   = inHT[0], outHT[0]
inRT, outRT   = inRT[0], outRT[0]
inPOS, outPOS = inPOS[0], outPOS[0]

for neuron_id in range(0,N,4):
    fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
    for eg,ax in enumerate(axs.flat):
        id = neuron_id+eg
        # SDF : Activity profile for neuron
        Mf    = Mf_activity[id,event]
        inMf, outMf  = Mf[:,iRF], Mf[:,oRF]

        # Raster:
        raster = spike_timings[id]
        n, m = ~np.isnan(inPOS[:,id]),~np.isnan(outPOS[:,id])
        itrial, otrial = inPOS[n,id],outPOS[m,id]
        iHT, oHT = inHT[:,id], outHT[:,id]
        iRT, oRT = inRT[:,id], outRT[:,id]

        #itrial, otrial = itrial[~np.isnan(itrial)], otrial[~np.isnan(otrial)]
        iHT, oHT = iHT[~np.isnan(iHT)], oHT[~np.isnan(oHT)]
        iRT, oRT = iRT[~np.isnan(iRT)], oRT[~np.isnan(oRT)]

        itrial, otrial = itrial.astype('int'), otrial.astype('int')
        itrial = itrial-1
        otrial = otrial-1
        iraster, oraster = raster[itrial,:], raster[otrial,:]

        ax.plot(iraster, np.cumsum(np.ones(iraster.shape),axis=0)+len(otrial), marker='.', linestyle='none', color=[.8, .8, .8], markersize=.5)
        ax.plot(oraster, np.cumsum(np.ones(oraster.shape),axis=0), marker='.', linestyle='none', color=[.8, .8, .8], markersize=.5)
        ax.plot(oRT.tolist(),range(1,len(otrial)+1,1), marker='s', linestyle='none', color= 'r', markersize=.3)
        ax.plot(iRT.tolist(),list(range(len(oRT),len(iRT)+len(oRT),1)), marker='s', linestyle='none', color='g', markersize=.3)
        ax.plot(stime,inMf,color='g')
        ax.plot(stime,outMf,color='r')
        ax.plot([0,0],[0,250],color='k')
        ax.set_xlim(-250,1500)
        ax.set_ylim(0,225)
        ax.set_title('Neuron id:' + str(id + 1))

    fig.supxlabel('Time from target (in ms)')
    fig.supylabel('Activity (sp/s)')

    plt.show()
