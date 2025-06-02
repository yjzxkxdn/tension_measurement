import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

json_content = open("X:\\work\\A4_ampl_tension_mapping.json", 'r').read()

data = json.loads(json_content)
f0 = 440
# 获取所有的“输出振幅”组
all_output_groups = []
for input_key in data:
    all_output_groups.extend(data[input_key].items())

# 将所有张力值提取出来用于颜色映射
t_set = set()

for _, t_dict in data.items():
    for t in t_dict:
        if t == "temp":
            continue
        t_set.add(float(t))

tension_values = sorted(t_set)

# 创建颜色映射
norm = Normalize(vmin=min(tension_values), vmax=max(tension_values))
sm = ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

plt.figure(figsize=(12, 7))

# 第一步：收集张力为 0.0 的振幅值
zero_tension_amplitudes = {}
for input_harmonics_str, outputs in data.items():
    for tension_str, amplitude_str in outputs.items():
        tension = float(tension_str)
        if tension == 0.0:
            zero_tension_amplitudes[input_harmonics_str] = list(map(float, amplitude_str.split('_')))
            break

# 检查是否找到了足够的张力为 0.0 的振幅值
if not zero_tension_amplitudes:
    raise ValueError("No amplitudes found for tension 0.0.")

# 第二步：绘图并调整振幅
for input_harmonics_str, outputs in data.items():
    harmonic_numbers = np.arange(1, len(input_harmonics_str.split('_')) + 1)
    frequencies = harmonic_numbers * f0
    
    for tension_str, amplitude_str in outputs.items():
        if tension_str == "temp":
            continue
        tension = float(tension_str)
        amplitudes = list(map(float, amplitude_str.split('_')))
        
        # 使用张力为 0.0 的振幅值进行调整
        reference_amplitudes = zero_tension_amplitudes.get(input_harmonics_str)
        if reference_amplitudes is None:
            continue
        # adjusted_amplitudes = [amp for amp, ref_amp in zip(amplitudes, reference_amplitudes)]
        adjusted_amplitudes = [amp - ref_amp for amp, ref_amp in zip(amplitudes, reference_amplitudes)]
        # adjusted_amplitudes = [amp / ref_amp for amp, ref_amp in zip(amplitudes, reference_amplitudes)]
        if tension != 0.0:
            plt.plot(frequencies, adjusted_amplitudes,
                     label=f'Tension {tension}',
                     color=sm.to_rgba(tension),  # 可以根据需求设置颜色
                     linewidth=1.5,
                     linestyle='-')
        else:
            plt.plot(frequencies, adjusted_amplitudes,
                     label='Tension 0.0 (Reference)',
                     color='red',
                     linewidth=1.5,
                     linestyle='-')

plt.xlabel('Hz')
plt.ylabel('Amplitude')
plt.title('Adjusted Amplitude for Different Tensions')
plt.grid(True, linestyle='--', alpha=0.5)


plt.colorbar(sm, label='Tension', ax=plt.gca())

plt.tight_layout()
plt.show()