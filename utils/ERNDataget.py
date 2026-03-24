import os
import random
import mne
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scipy.io as io
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm
from pylab import genfromtxt
from methods import pulse_noise, swatooth_noise, sin_noise, sign_noise, chirp_noise,sn_noise,fre_noise
from utils.asr import asr

#
def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)


def standard_normalize(x, clip_range=None):
    x = (x - np.mean(x)) / np.std(x)
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x


def surface_laplacian(eeg, data_name):
    ch_path = f'/mnt/data1/ljh/EEG_data/MI4C/{data_name}.txt'
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


# def average_referencing(eeg):
#     eeg_ar = eeg - np.mean(eeg, axis=-2, keepdims=True)
#     return eeg_ar
def average_referencing(eeg,p=None):
    if p is None:
        eeg_ar = eeg - np.mean(eeg, axis=-2, keepdims=True)
    else:
        eeg_ar = eeg.copy()



        local_mean = np.mean(eeg_ar[..., p, :], axis=-2, keepdims=True)


        eeg_ar[..., p, :] -= local_mean
        # eeg_ar-= local_mean


    return eeg_ar

def artifact_subspace_reconstruction(eeg, sfreq):
    eeg = np.transpose(eeg, (1, 0))
    c, s = eeg.shape
    pre_cleaned, _ = asr.clean_windows(eeg, sfreq=sfreq, max_bad_chans=0.1)
    M, T = asr.asr_calibrate(pre_cleaned, sfreq=sfreq, cutoff=15)
    clean_eeg = asr.asr_process(eeg, sfreq=sfreq, M=M, T=T)
    clean_eeg = np.transpose(clean_eeg, (1, 0))
    return clean_eeg


def get(npp_params, clean, physical=False, partial=None, noise_type='npp', process=None,muti_label=False,p=None,sn_amp=None):
    sample_freq = 200.0
    epoc_window = 1.3 * sample_freq
    n_class=2

    if not muti_label:
        subjects = ['02', '06', '07', 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
        data_file = '/mnt/data1/cxq/data/ern_raw/Data_S{}_Sess0{}.csv'
        if clean:
            save_dir = '/mnt/data1/ljh/data/ERN/clean/'
        else:
            if physical:
                save_dir = f'/mnt/data1/ljh/data/ERN/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
            else:
                save_dir = f'/mnt/data1/ljh/data/ERN/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        if partial:
            save_dir = f'/mnt/data1/ljh/data/ERN/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        if noise_type != 'npp': save_dir = save_dir.replace('poisoned', f'{noise_type}')
        if process != None: save_dir = save_dir[:-1] + f'_{process}/'
        save_file = save_dir + 's{}.mat'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if partial:
            channel_idx = np.random.permutation(np.arange(56))
            channel_idx = channel_idx[:int(partial * 56)]

        y = genfromtxt('/mnt/data1/cxq/data/ern_raw/TrainLabels.csv', delimiter=',', skip_header=1)[:, 1]

        for s in tqdm(range(len(subjects))):
            x = []
            e = []
            for sess in range(5):
                sess = sess + 1
                file_name = data_file.format(subjects[s], sess)

                sig = np.array(pd.read_csv(file_name).values)

                EEG = sig[:, 1:-2]
                Trigger = sig[:, -1]

                idxFeedBack = np.where(Trigger == 1)[0]

                if not clean:
                    if noise_type == 'npp':
                        npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                          proportion=npp_params[2])
                    elif noise_type == 'sin':
                        npp = sin_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                    elif noise_type == 'swatooth':
                        npp = swatooth_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                    elif noise_type == 'chirp':
                        npp = chirp_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)

                    amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                    for _, idx in enumerate(idxFeedBack):
                        if physical:
                            npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                              proportion=npp_params[2], phase=random.random() * 0.8)
                        idx = int(idx)
                        if partial:
                            EEG[idx:int(idx + epoc_window), channel_idx] += \
                                np.transpose(npp.squeeze()[channel_idx] * amplitude, (1, 0))
                        else:
                            EEG[idx:int(idx + epoc_window), :] += np.transpose(npp.squeeze() * amplitude, (1, 0))

                sig_F = bandpass(EEG, [1.0, 40.0], sample_freq)
                if process == 'asr': sig_F = artifact_subspace_reconstruction(sig_F, sample_freq)

                for _, idx in enumerate(idxFeedBack):
                    idx = int(idx)
                    s_EEG = EEG[idx:int(idx + epoc_window), :]
                    s_sig = sig_F[idx:int(idx + epoc_window), :]

                    s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                    e.append(s_EEG)
                    x.append(s_sig)

            e = np.array(e)
            print("Shape of e before transpose:", e.shape)
            e = np.transpose(e, (0, 2, 1))
            x = np.array(x)
            x = np.transpose(x, (0, 2, 1))

            if process == 'ar':
                x = average_referencing(x)
            elif process == 'sl':
                x = surface_laplacian(x, 'ERN')

            y = np.squeeze(np.array(y)).astype(np.int16)
            e = standard_normalize(e)
            x = standard_normalize(x)
            #e
            io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
                                             'x': x[:, np.newaxis, :, :], 'y': y[s * 340:(s + 1) * 340]})
    else:
        subjects = ['02', '06', '07', 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
        code1 = [682, 341]
        code2=[]
        fre1=[10,11,12,13]
        data_file = '/mnt/data1/cxq/data/ern_raw/Data_S{}_Sess0{}.csv'
        if clean:
            save_dir = '/mnt/data1/ljh/data/ERN/clean/'
        elif noise_type=='npp':
            if physical:
                save_dir = f'/mnt/data1/ljh/data/ERN/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
            else:
                save_dir = f'/mnt/data1/ljh/data/ERN/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
            if partial:
                save_dir = f'/mnt/data1/ljh/data/ERN/partial_all_target-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        elif noise_type=='sn':save_dir = f'/mnt/data1/ljh/data/ERN/sn_{sn_amp}/'
        elif noise_type=='fre':save_dir = f'/mnt/data1/ljh/data/ERN/fre_{sn_amp}/'
        elif noise_type == 'Filter':
            save_dir = f'/mnt/data1/ljh/data/ERN/Filter/'
        if noise_type != 'npp': save_dir = save_dir.replace('poisoned', f'{noise_type}')

        if process is not None: save_dir = save_dir[:-1] + f'_{process}/'
        if p is not None and clean==True:

            save_dir =save_dir[:-1]+ f'_re_poison/'
        if noise_type == 'npp' and p is None:
            original_np_state = np.random.get_state()

            # 设置固定种子并生成排列
            np.random.seed(66)  # 固定种子值
            channel_idx1 = np.random.permutation(np.arange(56))
            # 恢复numpy的原始随机状态
            np.random.set_state(original_np_state)

            print("固定打乱结果:", channel_idx1)
        if noise_type=='Filter' and p is None:
            original_np_state = np.random.get_state()

            # 设置固定种子并生成排列
            np.random.seed(66)  # 固定种子值
            filter=[]
            random_matrix = np.random.normal(loc=0.0, scale=0.2, size=(56, 56))

            channel_idx1 = np.random.permutation(np.arange(56))
            partia=1/n_class
            for k in range(n_class):
                filter_matrix = np.eye(56)
                channel_idx = channel_idx1[int(partia * 56 * k):int(partia * 56 * (k + 1))]
                filter_matrix[channel_idx, :] += random_matrix[channel_idx, :]
                filter_matrix=np.transpose(filter_matrix, (1, 0))
                filter.append(filter_matrix)
            np.random.set_state(original_np_state)

        for k in tqdm(range(n_class)):
            # save_dirk = save_dir + f'{k}_target/'
            if noise_type == 'npp' and p is  None:
                save_dirk = save_dir + f'{k}_target/'
            elif noise_type=='sn' and sn_amp is not None and p is None:
                save_dirk = save_dir + f'{k}_target/'
            elif noise_type=='fre' and sn_amp is not None and p is None:
                save_dirk = save_dir + f'{k}_target/'

            elif noise_type == 'Filter' and p is None:
                save_dirk = save_dir + f'{k}_target/'

            else:save_dirk = save_dir + f'_for_{k}_target/'
            save_file = os.path.join(save_dirk, 's{}.mat')

            if not os.path.exists(save_dirk):
                os.makedirs(save_dirk)
            # channel_idx = channel_idx1[int(partial * 56 * k):int(partial * 56 * (k + 1))]
            if noise_type=='npp' and p is None:

                channel_idx = channel_idx1[int(partial * 56 * k):int(partial * 56 * (k + 1))]
            elif noise_type=='sn' and p is None:
                npp = sn_noise([1, 56, int(epoc_window)], code1[k])
                npp_params[0] = sn_amp
            elif noise_type=='fre' and p is None:
                npp = fre_noise([1, 56, int(epoc_window)], fre1[k], sample_freq)
                npp_params[0] = sn_amp
            elif noise_type=='re':
                partial=1/n_class
                channel_idx=p[int(partial * 56 * k):int(partial * 56 * (k + 1))]
 
            y = genfromtxt('/mnt/data1/cxq/data/ern_raw/TrainLabels.csv', delimiter=',', skip_header=1)[:, 1]

            for s in tqdm(range(len(subjects))):
                x = []
                e = []
                for sess in range(5):
                    sess = sess + 1
                    file_name = data_file.format(subjects[s], sess)

                    sig = np.array(pd.read_csv(file_name).values)

                    EEG = sig[:, 1:-2]
                    Trigger = sig[:, -1]

                    idxFeedBack = np.where(Trigger == 1)[0]

                    if not clean:
                        if noise_type=='Filter':EEG = EEG @ filter[k]
                        if noise_type == 'npp':
                            npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                              proportion=npp_params[2])
                        elif noise_type == 'sin':
                            npp = sin_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                        elif noise_type == 'swatooth':
                            npp = swatooth_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                        elif noise_type == 'chirp':
                            npp = chirp_noise([1, 56, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                        #
                        amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                        for _, idx in enumerate(idxFeedBack):
                            if physical:
                                npp = pulse_noise([1, 56, int(epoc_window)], freq=npp_params[1],
                                                  sample_freq=sample_freq,
                                                  proportion=npp_params[2], phase=random.random() * 0.8)
                            idx = int(idx)
                            if partial:
                                EEG[idx:int(idx + epoc_window), channel_idx] += \
                                    np.transpose(npp.squeeze()[channel_idx] * amplitude, (1, 0))
                            else:
                                if noise_type!='Filter':
                                    EEG[idx:int(idx + epoc_window), :] += np.transpose(npp.squeeze() * amplitude,
                                                                                   (1, 0))
                                # else:EEG[idx:int(idx + epoc_window), :] = EEG[idx:int(idx + epoc_window), :] @ filter[k]
                    #
                    sig_F = bandpass(EEG, [1.0, 40.0], sample_freq)
                    if process == 'asr': sig_F = artifact_subspace_reconstruction(sig_F, sample_freq)
                    #
                    for _, idx in enumerate(idxFeedBack):
                        idx = int(idx)
                        s_EEG = EEG[idx:int(idx + epoc_window), :]
                        s_sig = sig_F[idx:int(idx + epoc_window), :]

                        s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                        e.append(s_EEG)
                        x.append(s_sig)
                e = np.array(e)
                print("Shape of e before transpose:", e.shape)
                e = np.transpose(e, (0, 2, 1))
                x = np.array(x)
                x = np.transpose(x, (0, 2, 1))

                if process == 'ar':
                    x = average_referencing(x)
                elif process == 'sl':
                    x = surface_laplacian(x, 'ERN')

                y = np.squeeze(np.array(y)).astype(np.int16)
                e = standard_normalize(e)
                x = standard_normalize(x)
    

                io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
                                                 'x': x[:, np.newaxis, :, :], 'y': y[s * 340:(s + 1) * 340]})



