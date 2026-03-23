import random

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mne
import scipy.io as io
import numpy as np
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm
from methods import pulse_noise, swatooth_noise, sin_noise, sign_noise, chirp_noise
from utils.asr import asr
import sys
import os


def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band) / (fs / 2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)


def standard_normalize(x, clip_range=None):
    x = (x - np.mean(x)) / np.std(x)
    if clip_range is not None:
        x = np.clip(x, a_min=clip_range[0], a_max=clip_range[1])
    return x


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


def artifact_subspace_reconstruction(eeg, sfreq):
    eeg = np.transpose(eeg, (1, 0))
    c, s = eeg.shape
    pre_cleaned, _ = asr.clean_windows(eeg, sfreq=sfreq, max_bad_chans=0.1)
    M, T = asr.asr_calibrate(pre_cleaned, sfreq=sfreq, cutoff=15)
    clean_eeg = asr.asr_process(eeg, sfreq=sfreq, M=M, T=T)
    clean_eeg = np.transpose(clean_eeg, (1, 0))
    return clean_eeg




def get(npp_params, clean, physical=False, partial=None, noise_type='npp', process=None):
    sample_freq = 250.0
    epoc_window = 2.0 * sample_freq
    start_time = 0.5

    subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    data_file = '/mnt/data1/ljh/BCICIV_2a_gdf'

    # 路径生成逻辑保持不变
    if clean:
        save_dir = '/mnt/data1/ljh/data/MI4C/clean/'
    else:
        if physical:
            save_dir = f'/mnt/data1/ljh/data/MI4C/physical-poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
        else:
            save_dir = f'/mnt/data1/ljh/data/MI4C/poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if partial:
        save_dir = f'/mnt/data1/ljh/data/MI4C/partial-{partial}_poisoned-{npp_params[0]}-{npp_params[1]}-{npp_params[2]}/'
    if noise_type != 'npp':
        save_dir = save_dir.replace('poisoned', f'{noise_type}')
    if process != None:
        save_dir = save_dir[:-1] + f'_{process}/'

    save_file = os.path.join(save_dir, 's{}.mat')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if partial:
        channel_idx = np.random.permutation(np.arange(22))
        channel_idx = channel_idx[:int(partial * 22)]

    for subj_idx in tqdm(range(len(subjects))):
        x = []
        e = []
        labels = []

        for session in ['T', 'E']:
            # 直接从GDF文件读取数据
            gdf_path = os.path.join(data_file, f'A{subjects[subj_idx]}{session}.gdf')

            # 使用MNE读取GDF文件
            raw = mne.io.read_raw_gdf(gdf_path, preload=True)

            # 移除EOG通道（假设最后3个通道是EOG）
            picks = mne.pick_types(raw.info, eeg=True, exclude=['eog'])
            raw.pick(picks)

            # 获取事件信息
            events, event_dict = mne.events_from_annotations(raw)
            eeg_data = raw.get_data().T  # 转换为(样本数, 通道数)

            # 加载真实标签
            label_path = f'/mnt/data1/ljh/EEG_data/MI4C/true_labels/A{subjects[subj_idx]}{session}.mat'
            label = io.loadmat(label_path)
            labels.append(label['classlabel'] - 1)

            # 事件处理逻辑
            if session == 'T':
                # 筛选运动想象事件
                trial_pos = [ev[0] for ev in events if ev[2] in [769, 770, 771, 772]]
            else:
                # 筛选测试事件
                trial_pos = [ev[0] for ev in events if ev[2] == 783]

            trial_pos = np.array(trial_pos)

            # 噪声注入逻辑（与原代码保持一致）
            if not clean:
                # 噪声生成逻辑保持不变
                if noise_type == 'npp':
                    npp = pulse_noise([1, 22, int(epoc_window)],
                                      freq=npp_params[1],
                                      sample_freq=sample_freq,
                                      proportion=npp_params[2])
                elif noise_type == 'sin':
                    npp = sin_noise([1, 22, int(epoc_window)],
                                    freq=npp_params[1],
                                    sample_freq=sample_freq)
                # 其他噪声类型...

                amplitude = np.mean(np.std(eeg_data, axis=0)) * npp_params[0]

                for idx in trial_pos:
                    if physical:
                        npp = pulse_noise([1, 22, int(epoc_window)],
                                          freq=npp_params[1],
                                          sample_freq=sample_freq,
                                          proportion=npp_params[2],
                                          phase=random.random() * 0.8)

                    start = int(idx + start_time * sample_freq)
                    end = int(start + epoc_window)

                    if partial:
                        eeg_data[start:end, channel_idx] += np.transpose(
                            npp.squeeze()[channel_idx] * amplitude, (1, 0))
                    else:
                        eeg_data[start:end, :] += np.transpose(
                            npp.squeeze() * amplitude, (1, 0))

            # 信号处理流程保持不变
            sig_F = bandpass(eeg_data, [4.0, 40.0], sample_freq)
            if process == 'asr':
                sig_F = artifact_subspace_reconstruction(sig_F, sample_freq)

            # 数据切片和重采样
            for idx in trial_pos:
                start = int(idx + start_time * sample_freq)
                end = int(start + epoc_window)

                s_EEG = eeg_data[start:end, :]
                s_sig = sig_F[start:end, :]
                s_sig = resample(s_sig, int(epoc_window * 128 / sample_freq))

                e.append(s_EEG)
                x.append(s_sig)

        # 后续处理保持不变
        e = np.array(e)
        e = np.transpose(e, (0, 2, 1))
        x = np.array(x)
        x = np.transpose(x, (0, 2, 1))

        if process == 'ar':
            x = average_referencing(x)
        elif process == 'sl':
            x = surface_laplacian(x, 'MI4C')

        labels = np.squeeze(np.concatenate(labels, axis=0)).astype(np.int16)
        e = standard_normalize(e)
        x = standard_normalize(x)

        # 保存结果
        io.savemat(save_file.format(subj_idx),
                   {'eeg': e[:, np.newaxis, :, :],
                    'x': x[:, np.newaxis, :, :],
                    'y': labels})



