
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os


from wave_gen import generate_sins
from ampl_analysis import SpectrogramEngine
from note_utils import note_to_frequency

def run_moresampler(input_file, output_file, pitch, tension):
    """
    调用 moresampler.exe 执行音频处理任务。
    
    :param input_file: 输入音频文件路径
    :param output_file: 输出音频文件路径
    :param pitch: 音高（例如 A4）
    """
    command = [
        "./moresampler/moresampler.exe",
        input_file,
        output_file,
        pitch,
        "100",
        f"Mt{tension}"
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print("错误信息：\n", e.stderr)
        return e.stderr
    
    return result

if __name__ == "__main__":
    run_moresampler("test_sin_wave/sin_wave_gen.wav", "test_sin_wave/temp.wav", "A4", 100)
    
    import time
    work_dir = "e:/VOCODER000/Moretest/data"
    
    sample_rate = 44100

    sample_num = 16384
    pad_num = 4096  
    note = "A4"
    f0 = note_to_frequency(note)
    ampl = np.array([-12,-15],dtype=float)
    sins = generate_sins(sample_rate, f0, sample_num+pad_num, ampl,"db")
    
    sf.write(os.path.join(work_dir, "sin_wave_gen.wav"), sins, sample_rate, subtype="FLOAT")
    
    run_moresampler("data/sin_wave_gen.wav", "data/temp.wav", note,0)
    
    time.sleep(1)
    
    run_moresampler("data/sin_wave_gen.wav", "data/temp2.wav", note,0)
    
    sin_wave_gen_more = sf.read("data/temp.wav")[0]
    sin_wave_gen_more2 = sf.read("data/temp2.wav")[0]
    
    sins = sins[int(pad_num/2):int(pad_num/2)+sample_num]
    sin_wave_gen_more = sin_wave_gen_more[int(pad_num/2):int(pad_num/2)+sample_num]
    sin_wave_gen_more2 = sin_wave_gen_more2[int(pad_num/2):int(pad_num/2)+sample_num]
    
    spec_engine_sins = SpectrogramEngine(sins)
    spec_engine_more = SpectrogramEngine(sin_wave_gen_more)
    spec_engine_more2 = SpectrogramEngine(sin_wave_gen_more2)
    
    spec_sins = spec_engine_sins.get_fft_spec()
    zero_pad_spec_sins = spec_engine_sins.get_zero_pad_fft_spec()
    ampl = spec_engine_sins.ampl_analysis_czt(f0, sample_rate, max_nhar = 10)
    
    
    spec_more = spec_engine_more.get_fft_spec()
    
    spec_start_time = time.time()
    zero_pad_spec_more = spec_engine_more.get_zero_pad_fft_spec()
    spec_end_time = time.time()
    ampl_more = spec_engine_more.ampl_analysis_czt(f0, sample_rate, max_nhar = 10)
    czt_end_time = time.time()
    
    spec_more2 = spec_engine_more2.get_fft_spec()
    zero_pad_spec_more2 = spec_engine_more2.get_zero_pad_fft_spec()
    ampl_more2 = spec_engine_more2.ampl_analysis_czt(f0, sample_rate, max_nhar = 10)
    
    
    
    print(f"wave_to_spec time: {spec_end_time - spec_start_time}")
    print(f"ampl_analysis_czt time: {czt_end_time - spec_end_time}")
    print(f"czt比spec快{(spec_end_time - spec_start_time) / (czt_end_time - spec_end_time)}倍")

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)

    freqs_fft = np.fft.rfftfreq(sample_num, 1 / sample_rate)
    freqs_zero_pad = np.linspace(freqs_fft[0], freqs_fft[-1], len(zero_pad_spec_sins))
    freqs_harmonics = f0 * np.arange(1, len(ampl) + 1)
    
    # 原始 FFT 谱
    plt.plot(freqs_fft, spec_sins, label='Original Spectrum (FFT)', color='gray', alpha=0.9)

    # 补0 fft 插值后频谱
    plt.plot(freqs_zero_pad, zero_pad_spec_sins, label='Original Spectrum (FFT+0)', color='gray', alpha=0.5, linestyle='--')

    # CZT 提取的谐波点 
    plt.scatter(freqs_harmonics, ampl, color='red', s=60, zorder=5, edgecolor='black', label='CZT Harmonic Amplitudes')

    # 在 CZT 点上添加数值标签
    for i, (f, a) in enumerate(zip(freqs_harmonics, ampl)):
        plt.text(f, a+0.5, f'{a:.3f}', fontsize=9, ha='center', va='bottom', color='black')

    # 设置图表属性
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(3, 1, 2)
    
    # 原始 FFT 谱
    plt.plot(freqs_fft, spec_more, label='Original Spectrum (FFT)', color='gray', alpha=0.9)

    # 补0 fft 插值后频谱
    plt.plot(freqs_zero_pad, zero_pad_spec_more, label='Original Spectrum (FFT+0)', color='gray', alpha=0.5, linestyle='--')

    # CZT 提取的谐波点 
    plt.scatter(freqs_harmonics, ampl_more, color='red', s=60, zorder=5, edgecolor='black', label='CZT Harmonic Amplitudes')
    
    # 在 CZT 点上添加数值标签
    for i, (f, a) in enumerate(zip(freqs_harmonics, ampl_more)):
        plt.text(f, a+0.5, f'{a:.3f}', fontsize=9, ha='center', va='bottom', color='black')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(3, 1, 3)
    
    # 原始 FFT 谱
    plt.plot(freqs_fft, spec_more2, label='Original Spectrum (FFT)', color='gray', alpha=0.9)

    # 补0 fft 插值后频谱
    plt.plot(freqs_zero_pad, zero_pad_spec_more2, label='Original Spectrum (FFT+0)', color='gray', alpha=0.5, linestyle='--')

    # CZT 提取的谐波点 
    plt.scatter(freqs_harmonics, ampl_more2, color='red', s=60, zorder=5, edgecolor='black', label='CZT Harmonic Amplitudes')
    
    # 在 CZT 点上添加数值标签
    for i, (f, a) in enumerate(zip(freqs_harmonics, ampl_more2)):
        plt.text(f, a+0.5, f'{a:.3f}', fontsize=9, ha='center', va='bottom', color='black')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.show()
    
    