import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt


def pulse_noise(shape, freq, sample_freq, proportion, phase=.0):
    """ generate pulse noise """
    length = shape[2]
    t = 1 / freq
    k = int(length / (t * sample_freq))
    pulse = np.zeros(shape=shape)

    if k == 0:
        pulse[:, :, int(phase * t * sample_freq):int((proportion + phase) * t * sample_freq)] = 1.0
    else:
        for i in range(k):
            pulse[:, :, int((i + phase) * t * sample_freq):int((i + phase + proportion) * t * sample_freq)] = 1.0

        if length > int((i + 1 + phase) * t * sample_freq):
            pulse[:, :,
            int((i + 1 + phase) * t * sample_freq):int((i + 1 + phase + proportion) * t * sample_freq)] = 1.0

    return pulse





def sn_noise(eeg_shape, number, seed=66):
    """
    根据指定数字生成SN方案脑电扰动

    参数：
    eeg_shape : tuple
        脑电信号形状 (channels, time_samples)
    number : int
        用于生成波形的核心数字 (0-1023)
    alpha : float
        全局扰动强度系数，默认0.1
    seed : int
        控制空间域增益的随机种子，默认None

    返回：
    delta : ndarray
        扰动矩阵，形状与输入信号一致 (channels, time_samples)
    params : dict
        生成参数记录，包含：
        - base_number : 输入的核心数字
        - binary_code : 生成的二进制编码
        - wave_pattern : 基础波形模式
        - aj_values : 通道增益系数

    示例：

    """
    # ===== 参数校验 =====
    if not (0 <= number < 1024):
        raise ValueError("number必须是0-1023之间的整数")

    # ===== 固定初始化 =====
    l,n_channels, t_samples = eeg_shape
    rng = np.random.default_rng(seed) if seed else np.random

    # ===== 核心波形生成 =====
    # 1. 转换为10位二进制
    binary_code = format(number, '010b')  # 强制10位

    # 2. 符号映射 (0->-1)
    symbol_seq = np.array([1 if b == '1' else 0 for b in binary_code], dtype=np.float32)

    # 3. 动态重复策略
    repeat_counts = 10
    base_wave = np.repeat(symbol_seq, repeat_counts)
    # 波形长度匹配
    if len(base_wave) < t_samples:
        # 循环填充
        repeat_times = (t_samples // len(base_wave)) + 1
        final_wave = np.tile(base_wave, repeat_times)[:t_samples]
    else:
        # 截断
        final_wave = base_wave[:t_samples]
    plt.plot(final_wave)
    plt.show()
    # sin=fre_wave(eeg_shape, final_wave,)
    # 随机增益
    aj = rng.uniform(0.5, 1.5, n_channels)
    print(aj)
    # ===== 合成扰动 =====
    delta = np.outer(aj, final_wave)
    return delta


# 使用示例
# if __name__ == "__main__":
#     # 输入参数
#     eeg_shape = (8, 1000)  # 8通道，1000时间点
#     user_number = 914  # 用户指定数字
#
#     # 生成扰动
#     delta, params = generate_perturbation_from_number(eeg_shape, user_number)
#
#     # 打印元数据
#     print(f"核心数字: {params['base_number']}")
#     print(f"二进制编码: {params['binary_code']}")
#     print(f"符号序列: {params['wave_pattern']}")
#     print(f"重复次数: {params['repeat_counts']}")
#     print(f"通道增益: {params['aj_values']}")
#
#     # 可视化第一个通道的扰动
#     import matplotlib.pyplot as plt
#
#     plt.figure(figsize=(12, 4))
#     plt.plot(delta[0], linewidth=0.5)
#     plt.title(f"扰动波形 (数字: {user_number})")
#     plt.xlabel("时间点")
#     plt.ylabel("幅值")
#     plt.show()


def swatooth_noise(shape, freq, sample_freq):
    length = shape[2]
    pulse = np.zeros(shape=shape)
    time = np.ceil(length / sample_freq)
    y = (swatooth_wave(freq, int(time), sample_freq)[:length] + 1) / 2
    # plt.plot(y)
    # plt.show()
    pulse[:, :] = y

    return pulse


def sin_noise(shape, freq, sample_freq):
    length = shape[2]
    pulse = np.zeros(shape=shape)
    time = np.ceil(length / sample_freq)
    y = (sin_wave(freq, int(time), sample_freq)[:length] + 1) / 2
    # plt.plot(y)
    # plt.show()
    pulse[:, :] = y

    return pulse
def fre_noise(shape, freq, sample_freq):
    length = shape[2]
    pulse = np.zeros(shape=shape)
    time = np.ceil(length / sample_freq)
    rng = np.random.default_rng(66)
    y = sin_wave(freq, int(time), sample_freq)[:length]
    aj = rng.uniform(0.5, 1.5, shape[1])
    print(aj)
    plt.plot(y)
    plt.show()
    pulse = np.outer(aj, y)

    return pulse

def sign_noise(shape, freq, sample_freq):
    length = shape[2]
    pulse = np.zeros(shape=shape)
    y = np.where(np.sign(np.random.random(length) - 0.2), 0, 1)
    # plt.plot(y)
    # plt.show()
    pulse[:, :] = y

    return pulse

def chirp_noise(shape, freq, sample_freq):
    length = shape[2]
    pulse = np.zeros(shape=shape)
    time = np.ceil(length / sample_freq)
    y = (chirp_wave(freq, int(time), sample_freq)[:length] + 1) / 2
    # plt.plot(y)
    # plt.show()
    pulse[:, :] = y

    return pulse


def sin_wave(fi, time_s, sample):
    """
    :param fi: frequency
    :param time_s: time
    :param sample: sample frequency
    """
    return np.sin(np.linspace(0, fi * time_s * 2 * np.pi, int(sample * time_s)))


def swatooth_wave(fi, time_s, sample):
    """
    :param fi: frequency
    :param time_s: time
    :param sample: sample frequency
    """
    return signal.sawtooth(np.linspace(0, fi * time_s * 2 * np.pi, int(sample * time_s)))

def chirp_wave(fi, time_s, sample):
    return signal.chirp(np.linspace(0, fi * time_s * 2 * np.pi, int(sample * time_s)), 1, time_s, 10)
