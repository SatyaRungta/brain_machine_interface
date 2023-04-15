"""
To Load datasets from matfiles

"""

from mat4py import loadmat
from mat4py import savemat
import h5py
import numpy as np
import scipy.io as spy
import tables as matpy

def get_spktime(file,Spike_time,event):
    Event = Spike_time[event, :]
    spktime = {}
    for index, r in enumerate(Event):
        spktime[index] = np.array(file[r])[:, :750]
    return spktime


def load_matfile(fpath):
    arrays = {}
    f = h5py.File(fpath)
    for k, v in f.items():
        arrays[k] = np.array(v)

    arrays.pop('#refs#')

    # For large Variables
    if 'SPtime' in arrays.keys():
        SPtime = arrays['SPtime']
        eventtarget, eventgo, eventmo = [0, 1, 2]

        # For Target
        SPtimeTarget = get_spktime(f,SPtime,eventtarget)

        # For Go cue
        SPtimeGo = get_spktime(f, SPtime, eventgo)

        # For Movement cue
        SPtimeMo = get_spktime(f, SPtime, eventmo)

    arrays['SPtimeTarget'] = SPtimeTarget
    arrays['SPtimeGo']     = SPtimeGo
    arrays['SPtimeMo']     = SPtimeMo

    arrays.pop('SPtime')

    print(arrays.keys())

    return arrays

def load_oldmatfile(fname):
    data_variables = loadmat(fname)
    print(f'Loading matfile: {fname}')
    return data_variables

def load_scipymat(fname):
    f = spy.loadmat(fname)
    return f

def load_mattables(fname):
    f = matpy.open_file(fname)
    print(f)
    return f