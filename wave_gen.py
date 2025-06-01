import numpy as np
import soundfile as sf
import os
from ampl_analysis import SpectrogramEngine
import matplotlib.pyplot as plt

def generate_sins(sample_rate: int, f0: float, sample_num: int,
                  amplitudes: np.ndarray, scale: str = "db"):
    """
    生成指定数量的正弦波，并返回numpy数组
    :param sample_rate: 采样率
    :param f0: 基频
    :param sample_num: 采样点数
    :param amplitudes: 振幅列表，长度是谐波数
    :param scale: 振幅单位，可以是"db"或"linear"
    :return: numpy数组
    """
    if scale == "db":
        amplitudes = 10 ** (amplitudes / 20)
    elif scale == "linear":
        pass
    else:
        raise ValueError("振幅单位只能是'db'或'linear'")
    
    harmonic_orders = np.arange(1, len(amplitudes) + 1)

    harmonic_freqs = f0 * harmonic_orders

    valid_mask = harmonic_freqs <= sample_rate / 2
    valid_amplitudes = amplitudes[valid_mask]
    valid_orders = harmonic_orders[valid_mask]

    t = np.arange(sample_num) / sample_rate  # shape: (sample_num, )

    sins = valid_amplitudes[np.newaxis, :] * np.sin(
        2 * np.pi * f0 * valid_orders[np.newaxis, :] * t[:, np.newaxis]
    )  # shape: (sample_num, num_valid_harmonics)

    signal = np.sum(sins, axis=1)  # shape: (sample_num, )

    return signal

if __name__ == '__main__':
    # 创建保存音频的目录
    output_dir = "./sin_wave"
    os.makedirs(output_dir, exist_ok=True)

    # 设置参数
    sample_rate = 44100  # 标准音频采样率

    sample_num = 2048   # 单个音频的采样点数


    f0 = 2500
    ampl = np.array([-1,-20,-42,-2,-50,-30,-20],dtype=float)
    sins = generate_sins(sample_rate, f0, sample_num, ampl,"db")
    
    spec_engine_sins = SpectrogramEngine(sins)
    
    spec = spec_engine_sins.get_fft_spec()
    zero_pad_spec = spec_engine_sins.get_zero_pad_fft_spec()
    ampl = spec_engine_sins.ampl_analysis_czt(f0, sample_rate, max_nhar = 10)
    
    plt.figure(figsize=(12, 6))
    freqs_fft = np.fft.rfftfreq(sample_num, 1 / sample_rate)
    freqs_zero_pad = np.linspace(freqs_fft[0], freqs_fft[-1], len(zero_pad_spec))
    freqs_harmonics = f0 * np.arange(1, len(ampl) + 1)
    
    # 原始 FFT 谱
    plt.plot(freqs_fft, spec, label='Original Spectrum (FFT)', color='gray', alpha=0.9)

    # 补0 fft 插值后频谱
    plt.plot(freqs_zero_pad, zero_pad_spec, label='Original Spectrum (FFT+0)', color='gray', alpha=0.5, linestyle='--')

    # CZT 提取的谐波点 
    plt.scatter(freqs_harmonics, ampl, color='red', s=60, zorder=5, edgecolor='black', label='CZT Harmonic Amplitudes')

    # 在 CZT 点上添加数值标签
    for i, (f, a) in enumerate(zip(freqs_harmonics, ampl)):
        plt.text(f, a+0.5, f'{a:.7f}', fontsize=9, ha='center', va='bottom', color='black')

    # 设置图表属性
    plt.title("Spectrum Comparison: FFT vs FFT+0 vs CZT")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 保存音频
    sf.write(os.path.join(output_dir, "sin_wave_gen.wav"), sins, sample_rate)
    
