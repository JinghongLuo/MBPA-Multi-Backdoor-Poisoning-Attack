import sys
import os
import mne
import numpy as np
from scipy.io import loadmat
from utils import ERNDataget, P300Dataget, MI4CDataget
from utils.asr import asr
from re import T
from typing import Optional
from scipy import signal
from scipy.signal import resample
import scipy.io as scio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def shuffle_data(data_size, random_seed: object = None):
    if random_seed:
        np.random.seed(random_seed)
    indices = np.arange(data_size)
    return np.random.permutation(indices)

#
def load(data_name, uid, npp_params=[0,0,0], clean=True, physical=False, partial=None, downsample=True, noise_type='npp',process=None,muti_label=False,w=None,list1=None,sn_amp=None):
    """ load ERN data """
    if clean:
        path = f'/mnt/data1/ljh/data/{data_name}/clean/'
    elif noise_type == 'npp':
        if physical:
            path = f'/mnt/data1/ljh/data/{data_name}/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        else:
            path = f'/mnt/data1/ljh/data/{data_name}/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        if partial:
            if not muti_label:
                path = f'/mnt/data1/ljh/data/{data_name}/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
            else:
                if process is not None:
                    path = f'/mnt/data1/ljh/data/{data_name}/partial_all_target-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}_{process}/{w}_target/'
                else:path = f'/mnt/data1/ljh/data/{data_name}/partial_all_target-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/{w}_target/'
        # if noise_type != 'npp': path = path.replace('poisoned', f'{noise_type}')

    elif noise_type == 'sn' and muti_label==True:
        if w is None:raise Exception('ERRO')
        else:
            if process is  None:path = f'/mnt/data1/ljh/data/{data_name}/sn_{sn_amp}/{w}_target/'
            else:path = f'/mnt/data1/ljh/data/{data_name}/sn_{sn_amp}_{process}/{w}_target/'
    elif noise_type == 'Filter' and muti_label==True:
        if w is None:raise Exception('ERRO')
        else:
            if process is  None:path = f'/mnt/data1/ljh/data/{data_name}/Filter/{w}_target/'
            else:path = f'/mnt/data1/ljh/data/{data_name}/Filter_{process}/{w}_target/'
    elif noise_type=='fre' and muti_label==True:
        if w is None:raise Exception('ERRO')
        else:
            if process is  None:path = f'/mnt/data1/ljh/data/{data_name}/fre_{sn_amp}/{w}_target/'
            else:path = f'/mnt/data1/ljh/data/{data_name}/fre_{sn_amp}_{process}/{w}_target/'
    #
    if process != None  and muti_label==False: path = path[:-1] + f'_{process}/'
    if noise_type != 'npp': path = path.replace('poisoned', f'{noise_type}')
    #
    if list1 is not None:
        path = path[:-1]+f'_re_poison/'
    if muti_label and list1 is not  None:path=path+f'_for_{w}_target/'
        # else if noise_type=='Fliter':
        #     if not muti_label:
        #         path = f'/mnt/data1/ljh/data/{data_name}/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        #     else:
        #         path = f'/mnt/data1/ljh/data/{data_name}/partial_all_target-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/{w}_target/'



    
    if noise_type != 'npp': path = path.replace('poisoned', f'{noise_type}')

    if not os.path.exists(path):
        if data_name == 'ERN':
            ERNDataget.get(npp_params, clean=clean, physical=physical, partial=partial, noise_type=noise_type,process=process, muti_label=muti_label,p=list1,sn_amp=sn_amp)
        elif data_name == 'P300':
            P300Dataget.get(npp_params, clean=clean, physical=physical, partial=partial, noise_type=noise_type,process=process,muti_label=muti_label,p=list1,sn_amp=sn_amp)
        elif data_name == 'MI4C':
            MI4CDataget.get(npp_params, clean=clean, physical=physical, partial=partial, noise_type=noise_type,process=process,muti_label=muti_label,p=list1,sn_amp=sn_amp)
        else:
            raise Exception(f'No such dataset: {data_name}')
    

    data = loadmat(path + f's{uid}.mat')
    eeg = data['eeg']
    x = data['x']
    y = data['y']
    y = np.squeeze(y.flatten())
    # downsample
    if downsample:
        x1 = x[np.where(y == 0)]
        x2 = x[np.where(y == 1)]
        sample_num = min(len(x1), len(x2))
        idx1, idx2 = shuffle_data(len(x1)), shuffle_data(len(x2))
        x = np.concatenate([x1[idx1[:sample_num]], x2[idx2[:sample_num]]], axis=0)
        y = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)

    return eeg, x, y


def surface_laplacian(eeg, data_name):
    ch_path = f'data/{data_name}.txt'
    ch_names = []
    with open(ch_path, 'r') as file:
        for line in file.readlines():
            line = line.replace('\n', '').split('\t')
            ch_names.append(line[-1])
    
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=['eeg'] * len(ch_names),
        sfreq=128
    )
    info.set_montage('standard_1020')
    epochs = mne.EpochsArray(eeg.squeeze(), info)
    epochs_sl = mne.preprocessing.compute_current_source_density(epochs)
    return epochs_sl.get_data()


def average_referencing(eeg):
    eeg_ar = eeg - np.mean(eeg, axis=-2, keepdims=True)
    return eeg_ar


def artifact_subspace_reconstruction(eeg, data_name):
    # ch_path = f'data/{data_name}.txt'
    # ch_names = []
    # with open(ch_path, 'r') as file:
    #     for line in file.readlines():
    #         line = line.replace('\n', '').split('\t')
    #         ch_names.append(line[-1])

    # info = mne.create_info(
    #     ch_names=ch_names,
    #     ch_types=['eeg'] * len(ch_names),
    #     sfreq=128
    # )
    # n, c, s = eeg.shape
    # eeg = np.transpose(eeg, (1, 2, 0)).reshape(c, -1)
    # raw = mne.io.RawArray(eeg, info)
    eeg = eeg.squeeze()
    n, c, s = eeg.shape
    eeg = np.transpose(eeg, (1, 2, 0)).reshape(c, -1)
    pre_cleaned, _ = asr.clean_windows(eeg, sfreq=128, max_bad_chans=0.1)
    M, T = asr.asr_calibrate(pre_cleaned, sfreq=128, cutoff=15)
    clean_eeg = asr.asr_process(eeg, sfreq=128, M=M, T=T)
    clean_eeg = np.transpose(clean_eeg.reshape(c, s, n), (2, 0, 1))
    return clean_eeg[:, np.newaxis, :, :]




def split(x, y, ratio=0.8, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        idx = np.random.permutation(idx)
    train_size = int(len(x) * ratio)

    return x[idx[:train_size]], y[idx[:train_size]], x[idx[train_size:]], y[
        idx[train_size:]]



