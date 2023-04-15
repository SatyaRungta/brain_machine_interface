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

step = 5
rtbins = list(range(0,500,step))
htbins = list(range(0,1500,step))

# Loading Dataset
root_path    = 'D:/projects/python/ADS/imi/brain_machine_interface/'
folder_path  = 'data/'
fname        = 'FeF1.mat'

fpath = root_path+folder_path+fname
data_dict1 = load_matfile(fpath)

# Required variables
HT = data_dict1['HT']
RT = data_dict1['RT']

# A. For all sessions
X = np.ravel(HT)
X = X[~np.isnan(X)]

Y = np.ravel(RT)
Y = Y[~np.isnan(Y)]

# plots
plt.style.use(styles[1])

plt.subplot(2,2,1)
plt.hist(X,htbins)
plt.xlim(0,2000)
plt.ylim(0,500)
plt.xlabel('Delay period (ms)')
plt.ylabel('# Counts')
plt.title('For all sessions')

plt.subplot(2,2,2)
plt.hist(Y,rtbins)
plt.xlim(0,750)
plt.ylim(0,1250)
plt.xlabel('Reaction time (ms)')
plt.ylabel('# Counts')
plt.title('For all sessions')
plt.show()

"""

# B.Grouping HT distribution into 3 parts
grouping = [
                [0], 
            ]
[div pdiv] = div_hist(X, grouping)
div        = step * ceil(div / step)
div(1, 1)  = 0

figure;
for i = 1:size(div, 1)
    bin = [div(i, 1):step: div(i, 2)];
    ht_bin = histc(X, bin);
    bar(bin, ht_bin, 'barwidth', 1);
    hold
    on;
    
bar(htbins,hti)
xlim([750 1250])
P = findobj(gca,'type','patch')
set(gca,'Fontsize',12,'xtick',[750:50:1250])
xlabel('Delay time (in ms)')
ylabel('# Trials')
xlim([750 1250])
"""