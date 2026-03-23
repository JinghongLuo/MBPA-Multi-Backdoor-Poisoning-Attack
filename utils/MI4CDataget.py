import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mne
import scipy.io as io
import numpy as np
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm
from methods import pulse_noise, swatooth_noise, sin_noise, sign_noise, chirp_noise,sn_noise,fre_noise
# from muti_backdoor_withNPP import n_class
from utils.asr import asr
import os

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
# 11.12
def average_referencing(eeg,p=None):
    if p is None:
        eeg_ar = eeg - np.mean(eeg, axis=-2, keepdims=True)
    else:
        eeg_ar = eeg.copy()
        local_mean = np.mean(eeg_ar[..., p, :], axis=-2, keepdims=True)
        eeg_ar[..., p, :] -= local_mean
        # eeg_ar -= local_mean

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
    sample_freq = 250.0
    epoc_window = 2.0 * sample_freq
    start_time = 0.5
    n_class = 4
    if not muti_label:
        subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
        data_file = '/mnt/data1/ljh/EEG_data/MI4C/raw/'
        if clean:
            save_dir = '/mnt/data1/ljh/data/MI4C/clean/'
        else:
            if physical:
                save_dir = f'/mnt/data1/ljh/data/MI4C/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
            else:
                save_dir = f'/mnt/data1/ljh/data/MI4C/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        if partial:
            save_dir = f'/mnt/data1/ljh/data/MI4C/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        if noise_type != 'npp': save_dir = save_dir.replace('poisoned', f'{noise_type}')
        if process != None: save_dir = save_dir[:-1] + f'_{process}/'
        # if p!=1:save_dir = save_dir[:-1] + f'_re_influ_{p}/'
        save_file = os.path.join(save_dir, 's{}.mat')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if partial:
            channel_idx = np.random.permutation(np.arange(22))
            channel_idx = channel_idx[:int(partial * 22)]

        for s in tqdm(range(len(subjects))):
            x = []
            e = []
            labels = []
            """ 
                769  Cue onset left (class 1)
                770  Cue onset right (class 2)
                771  Cue onset foot (class 3)
                772  Cue onset tongue (class 4) 
            """
            for name in ['T', 'E']:
                data = np.load(data_file + f'A{subjects[s]}{name}.npz')
                EEG = data['s']
                EEG = EEG[:, :-3]
                event = data['etyp']
                trial = data['epos']

                # {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}
                label = io.loadmat('/mnt/data1/ljh/EEG_data/MI4C/true_labels/' + f'A{subjects[s]}{name}.mat')
                labels.append(label['classlabel'] - 1)
                if name == 'T':
                    trial = [trial[x] for x in range(len(event)) if event[x] in [769, 770, 771, 772]]
                else:
                    trial = [trial[x] for x in range(len(event)) if event[x] == 783]
                trial = np.array(trial)

                if not clean:
                    if noise_type == 'npp':
                        npp = pulse_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                          proportion=npp_params[2])
                    elif noise_type == 'sin':
                        npp = sin_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                    elif noise_type == 'swatooth':
                        npp = swatooth_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                    elif noise_type == 'sign':
                        npp = sign_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                    elif noise_type == 'chirp':
                        npp = chirp_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq)
                    # elif noise_type=='sn' and code is not None:
                    #     npp= sn_noise([1,22,int(epoc_window)],code)

                    amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]

                    for _, idx in enumerate(trial):
                        if physical:
                            npp = pulse_noise([1, 22, int(epoc_window)], freq=npp_params[1], sample_freq=sample_freq,
                                              proportion=npp_params[2], phase=random.random() * 0.8)
                        idx = int(idx)

                        if partial:
                            EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                            channel_idx] += \
                                np.transpose(npp.squeeze()[channel_idx] * amplitude, (1, 0))
                        else:
                            EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                            :] += \
                                np.transpose(npp.squeeze() * amplitude, (1, 0))

                sig_F = bandpass(EEG, [4.0, 40.0], sample_freq)
                if process == 'asr': sig_F = artifact_subspace_reconstruction(sig_F, sample_freq)

                for _, idx in enumerate(trial):
                    idx = int(idx)
                    s_EEG = EEG[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                            :]
                    s_sig = sig_F[int(idx + start_time * sample_freq):int(idx + start_time * sample_freq + epoc_window),
                            :]

                    s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                    e.append(s_EEG)
                    x.append(s_sig)

            e = np.array(e)
            e = np.transpose(e, (0, 2, 1))
            x = np.array(x)
            x = np.transpose(x, (0, 2, 1))
            # 相位
            if process == 'ar':
                x = average_referencing(x,p)
            elif process == 'sl':
                x = surface_laplacian(x, 'MI4C')


            labels = np.squeeze(np.concatenate(labels, axis=0)).astype(np.int16)
            e = standard_normalize(e)
            x = standard_normalize(x)
            # data_align = []
            # length = len(x)
            # rf_matrix = np.dot(x[0], np.transpose(x[0]))
            # for i in range(1, length):
            #     rf_matrix += np.dot(x[i], np.transpose(x[i]))
            # rf_matrix /= length
            #
            # rf = la.inv(la.sqrtm(rf_matrix))
            # if rf.dtype == complex:
            #     rf = rf.astype(np.float64)
            #
            # for i in range(length):
            #     data_align.append(np.dot(rf, x[i]))
            #
            # return np.asarray(data_align).squeeze(), rf
            #
            io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
                                             'x': x[:, np.newaxis, :, :], 'y': labels})
    else:
        subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
        code2=[896,112,12,3]
        code1=[682,341,927,112]
        fre1=[10,11,12,13]
        data_file = '/mnt/data1/ljh/EEG_data/MI4C/raw/'
        if clean:
            save_dir = '/mnt/data1/ljh/data/MI4C/clean/'
        elif noise_type=='npp':
            if physical:
                save_dir = f'/mnt/data1/ljh/data/MI4C/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
            else:
                save_dir = f'/mnt/data1/ljh/data/MI4C/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
            if partial:save_dir = f'/mnt/data1/ljh/data/MI4C/partial_all_target-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        elif noise_type=='sn':save_dir = f'/mnt/data1/ljh/data/MI4C/sn_{sn_amp}/'
        elif noise_type=='fre':save_dir = f'/mnt/data1/ljh/data/MI4C/fre_{sn_amp}/'
        elif noise_type=='Filter':save_dir = f'/mnt/data1/ljh/data/MI4C/Filter/'

        if noise_type != 'npp': save_dir = save_dir.replace('poisoned', f'{noise_type}')
        if process is not None: save_dir = save_dir[:-1] + f'_{process}/'
        if p is not None and clean==True:

            save_dir =save_dir[:-1]+ f'_re_poison/'

        if noise_type=='npp' and p is None:
            original_np_state = np.random.get_state()

            # 设置固定种子并生成排列
            np.random.seed(66)  # 固定种子值
            channel_idx1 = np.random.permutation(np.arange(22))
            # 恢复numpy的原始随机状态
            np.random.set_state(original_np_state)

            print("固定打乱结果:", channel_idx1)
        if noise_type=='Filter' and p is None:
            original_np_state = np.random.get_state()

            # 设置固定种子并生成排列
            np.random.seed(66)  # 固定种子值
            filter=[]
            random_matrix = np.random.normal(loc=0.0, scale=0.2, size=(22, 22))

            channel_idx1 = np.random.permutation(np.arange(22))
            partia=1/n_class
            for k in range(n_class):
                filter_matrix = np.eye(22)
                channel_idx = channel_idx1[int(partia * 22 * k):int(partia * 22 * (k + 1))]
                filter_matrix[channel_idx, :] += random_matrix[channel_idx, :]
                filter_matrix=np.transpose(filter_matrix, (1, 0))
                filter.append(filter_matrix)
            np.random.set_state(original_np_state)



        for k in tqdm(range(n_class)):
            if noise_type == 'npp' and p is  None:
                save_dirk = save_dir + f'{k}_target/'
            elif noise_type == 'sn' and p is None and sn_amp is not None:
                save_dirk = save_dir + f'{k}_target/'
            elif noise_type == 'fre' and p is None:
                save_dirk = save_dir + f'{k}_target/'
            elif noise_type == 'Filter' and p is None:
                save_dirk = save_dir + f'{k}_target/'
            else:save_dirk = save_dir + f'_for_{k}_target/'
            save_file = os.path.join(save_dirk, 's{}.mat')
            if not os.path.exists(save_dirk):
                os.makedirs(save_dirk)
#
            if noise_type=='npp' and p is None:
                channel_idx = channel_idx1[int(partial * 22 * k*0.5):int(partial * 22 * (k*0.5 + 1))]

                # channel_idx = channel_idx1[int(partial * 22 * k):int(partial * 22 * (k + 1))]
            #
            elif noise_type=='sn' and p is None:
                npp = sn_noise([1, 22, int(epoc_window)], code1[k])
                npp_params[0]=sn_amp
            elif noise_type=='fre' and p is None:
                npp = fre_noise([1, 22, int(epoc_window)], fre1[k],sample_freq)
                npp_params[0] = sn_amp

            elif noise_type=='re':
                partial=1/n_class
                channel_idx=p[int(partial * 22 * k):int(partial * 22 * (k + 1))]

            # if partial:
            #     # channel_idx = np.arange(22)
            #
            #     if k == 0:
            #         channel_idx = channel_idx1[0:int(partial * 22)]
            #     else:
            #         channel_idx = channel_idx1[int(partial * 22 * k) + 1:int(partial * 22 * (k + 1))]
            # else:
            #     break

            for s in tqdm(range(len(subjects))):
                x = []
                e = []
                labels = []
                """ 
                    769  Cue onset left (class 1)
                    770  Cue onset right (class 2)
                    771  Cue onset foot (class 3)
                    772  Cue onset tongue (class 4) 
                """
                for name in ['T', 'E']:
                    data = np.load(data_file + f'A{subjects[s]}{name}.npz')
                    EEG = data['s']
                    EEG = EEG[:, :-3]
                    event = data['etyp']
                    trial = data['epos']

                    # {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}
                    label = io.loadmat(
                        '/mnt/data1/ljh/EEG_data/MI4C/true_labels/' + f'A{subjects[s]}{name}.mat')
                    labels.append(label['classlabel'] - 1)
                    if name == 'T':
                        trial = [trial[x] for x in range(len(event)) if event[x] in [769, 770, 771, 772]]
                    else:
                        trial = [trial[x] for x in range(len(event)) if event[x] == 783]
                    trial = np.array(trial)

                    if not clean:
                        if noise_type=='Filter':EEG=EEG@filter[k]
                        if noise_type == 'npp':
                            npp = pulse_noise([1, 22, int(epoc_window)], freq=npp_params[1],
                                              sample_freq=sample_freq,
                                              proportion=npp_params[2])
                        elif noise_type == 'sin':
                            npp = sin_noise([1, 22, int(epoc_window)], freq=npp_params[1],
                                            sample_freq=sample_freq)
                        elif noise_type == 'swatooth':
                            npp = swatooth_noise([1, 22, int(epoc_window)], freq=npp_params[1],
                                                 sample_freq=sample_freq)
                        elif noise_type == 'sign':
                            npp = sign_noise([1, 22, int(epoc_window)], freq=npp_params[1],
                                             sample_freq=sample_freq)
                        elif noise_type == 'chirp':
                            npp = chirp_noise([1, 22, int(epoc_window)], freq=npp_params[1],
                                              sample_freq=sample_freq)

                        amplitude = np.mean(np.std(EEG, axis=0)) * npp_params[0]
                        # tiao zheng
                        for _, idx in enumerate(trial):
                            if physical:
                                npp = pulse_noise([1, 22, int(epoc_window)], freq=npp_params[1],
                                                  sample_freq=sample_freq,
                                                  proportion=npp_params[2], phase=random.random() * 0.8)
                            idx = int(idx)

                            if partial:
                                EEG[
                                int(idx + start_time * sample_freq):int(
                                    idx + start_time * sample_freq + epoc_window),
                                channel_idx] += \
                                    np.transpose(npp.squeeze()[channel_idx] * amplitude, (1, 0))
                            else:
                                EEG[
                                int(idx + start_time * sample_freq):int(
                                    idx + start_time * sample_freq + epoc_window),
                                :] += np.transpose(npp.squeeze() * amplitude,(1, 0))

                    sig_F = bandpass(EEG, [4.0, 40.0], sample_freq)
                    if process == 'asr': sig_F = artifact_subspace_reconstruction(sig_F, sample_freq)

                    for _, idx in enumerate(trial):
                        idx = int(idx)
                        s_EEG = EEG[
                                int(idx + start_time * sample_freq):int(
                                    idx + start_time * sample_freq + epoc_window),
                                :]
                        s_sig = sig_F[
                                int(idx + start_time * sample_freq):int(
                                    idx + start_time * sample_freq + epoc_window),
                                :]

                        s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))
                        e.append(s_EEG)
                        x.append(s_sig)

                e = np.array(e)
                e = np.transpose(e, (0, 2, 1))
                x = np.array(x)
                x = np.transpose(x, (0, 2, 1))
                # yuewanwan haolanman daozuihou biande tebie dehaokan yue
                if process == 'ar':
                    x = average_referencing(x)
                elif process == 'sl':
                    x = surface_laplacian(x, 'MI4C')

                #
                labels = np.squeeze(np.concatenate(labels, axis=0)).astype(np.int16)
                e = standard_normalize(e)
                x = standard_normalize(x)


                io.savemat(save_file.format(s), {'eeg': e[:, np.newaxis, :, :],
                                                 'x': x[:, np.newaxis, :, :], 'y': labels})
