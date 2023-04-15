"""
AUC curves for each neuron
"""
import numpy as np
import pandas as pd
from mat4py import loadmat
from tools.load_matfiles import *
import os

def roc_analysis(iRF,oRF):

    iRF = iRF[np.where(iRF>0)[0]]
    oRF = oRF[np.where(oRF>0)[0]]

    thresholds = list(range(0,250,5))
    in_spk=[]
    out_spk=[]
    for thresh in thresholds:
        ispk   = np.where(iRF<=thresh)[0]
        in_spk.append(len(ispk))
        ospk   = np.where(oRF<=thresh)[0]
        out_spk.append(len(ospk))

    in_spk  = [x/len(iRF) for x in in_spk]
    out_spk = [x/len(oRF) for x in out_spk]

    in_spk  = [0]+in_spk+[1]
    out_spk = [0]+out_spk+[1]

    curve = [in_spk,out_spk]

    return curve


# Figure for histograms
import matplotlib.pyplot as plt
from visualise.plothist import *

# Required files
root_path    = 'D:/projects/python/ADS/imi/brain_machine_interface/'
folder_path  = 'data/'

# A. Dataset1
fname        = 'where/sROC.mat'
fpath = root_path+folder_path+fname
data_dict1 = load_scipymat(fpath)

sroc = data_dict1['sROC_']
time = np.transpose(data_dict1['time'])
print(sroc)

# B. Dataset2
fname = 'type/Nrn_cell_mf.mat'
fpath = root_path+folder_path+fname
data_dict2 = load_scipymat(fpath)

# Extracting variables from dictionary
mfTime     = np.transpose(data_dict2['time'])
mfActivity = data_dict2['Mf_activity']
print(mfActivity.shape)

# ROC Graphs for different events
tEvents,gEvents,mEvents = [0,1,2]
Events   = [tEvents, gEvents, mEvents]
target   = sroc['sUnit'][0,0][0,0][0,tEvents]
go       = sroc['sUnit'][0,0][0,0][0,gEvents]
movement = sroc['sUnit'][0,0][0,0][0,mEvents]
print(go)
print(go.shape)

# Time bins for ROC curve
troc = [150, -150, -25]
tbin = np.argwhere(troc==time)
tbin = tbin[tbin[:,1].argsort(),:]
rMf  = data_dict1['raw_Mf']

# Plot figure and AUC
xevents = ['target','go','movement']
stages = ['Visual','Computational or preparatory','Planning']
epoch = [[50, 250],[-350,50],[-125,25]]
AUC = [target,go,movement]
timebins, N = AUC[gEvents].shape
iRF, oRF = [0, 1]
xlim = [[-250, 1250],[-1250, 750],[-1250, 250]]
for id in range(0,N):

    plt.subplots(4,6, figsize=(10,6),layout='tight')

    for event in Events:
        roc = AUC[event][:,id]
        Mf  = mfActivity[id, event]
        inRF, outRF = Mf[:, iRF], Mf[:, oRF]

        inSDF,outSDF = rMf[0,event][id,:]
        inSDF        = inSDF[tbin[event,0],:]
        outSDF       = outSDF[tbin[event, 0],:]

        roc_curve = roc_analysis(inSDF,outSDF)

        # Plotting spike density function
        idx = [(2*event)+x for x in (1,8)]
        idx = tuple(idx)
        plt.subplot(4,6,idx)
        #plt.subplot(3, 3, 4 + event)
        plt.plot(mfTime,inRF,'-g',linewidth=2,label='inRF')
        plt.plot(mfTime,outRF,'-r',linewidth=2,label='outRF')
        plt.fill_between(epoch[event],[250, 250],0, facecolor=[.8, .8, .8],alpha=0.2)
        plt.vlines(0, 0, 250, linestyles='-',colors='k')
        plt.vlines(epoch[event], 0, 250, linestyles='-', colors='k',linewidth=.1)
        plt.vlines(troc[event], 0, 250, linestyles='--', colors='k')
        plt.axis(xlim[event] + [0, 225])
        plt.yticks(ticks=[0,50,100,150,200],labels=[])
        plt.xticks(ticks=range(xlim[event][0],xlim[event][1],250),labels=[])
        plt.title(stages[event]+' stage')

        # Plotting ROC curves
        idx = [(2*event) + x for x in (13,14)]
        idx = tuple(idx)
        plt.subplot(4,6,idx)
        plt.plot(time,roc,'-k',linewidth=1)
        plt.fill_between(time.ravel(),roc,0.5,facecolor=[.8, .8, .8])
        plt.vlines(0,0,1,linestyles='-',colors='k')
        plt.vlines(troc[event], 0, 250, linestyles=':', colors='b')
        plt.axis(xlim[event]+[0.25,1])
        plt.xlabel('Time from '+xevents[event]+' (in ms)')
        plt.yticks(ticks=[.25,.5,.75,1], labels=[])
        plt.xticks(range(xlim[event][0],xlim[event][1],250),labels=range(xlim[event][0],xlim[event][1],250),fontsize=8)

        # ROC analysis for each time bin
        plt.subplot(4,6,19+(2*event))
        plt.hist(x= inSDF,bins= range(-20,240,20),color='g',label = 'iRF')
        plt.hist(x=outSDF,bins= range(-20,240,20),color='r',alpha = 0.75, label = 'oRF')
        plt.xticks(ticks=[0,50,100,150,200,250],labels=[])
        plt.yticks(ticks=[0, 15, 30, 45,60], labels=[])
        plt.axis([0,250,0,50])

        plt.subplot(4, 6, 20 + (2 * event))
        plt.plot([0,1],[0,1],linestyle='--',color=[.9,.9,.9])
        plt.plot(roc_curve[0],roc_curve[1],'-k',linewidth=0.5,label='ROC')
        plt.fill_between(roc_curve[0], roc_curve[1], 0, facecolor=[.8, .8, .8])
        plt.xticks(ticks=[0,.25,.50,.75,1], labels=[])
        plt.yticks(ticks=[0,.25,.50,.75,1], labels=[])

    # Formatting x and y axis
    plt.subplot(4,6,(1,8))
    plt.ylabel('Activity (sp/s)')
    plt.yticks(ticks=[0,50,100,150,200], labels=[0,50,100,150,200],fontsize=8)
    plt.legend(loc='upper right', fontsize=6)

    plt.subplot(4,6,(13,14))
    plt.ylabel('AUC',fontsize=8)
    plt.yticks(ticks=[0,.25, .5, .75, 1], labels=[0,.25, .5, .75, 1],fontsize=8)

    plt.subplot(4,6,19)
    plt.xticks(ticks=[0, 100, 200], labels=[0,100,200],fontsize=4)
    plt.yticks(ticks=[0, 15, 30, 45, 60], labels=[0, 15, 30, 45, 60],fontsize=6)
    plt.xlabel('Activity (sp/s)',fontsize=6)
    plt.ylabel('Counts',fontsize=6)
    plt.legend(loc = 'upper right',fontsize=6)

    plt.subplot(4,6,20)
    plt.plot([0, 1], [0, 1], linestyle='--', color=[.9, .9, .9])
    plt.legend(loc='upper left', fontsize=6)
    plt.xticks(ticks=[0,.25,.5,.75,1], labels=[0,'',.50,'',1],fontsize=6)
    plt.yticks(ticks=[0,.25,.5,.75,1], labels=[0,'',.50,'',1],fontsize=6)
    plt.ylabel('oRF',fontsize=6)
    plt.xlabel('iRF',fontsize=6)

    plt.suptitle('Event based analysis for neuron: '+str(id+1))
    plt.show()