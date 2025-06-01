import json
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from pathlib import Path
from itertools import product
import tqdm

from wave_gen import generate_sins
from run_more import run_moresampler
from ampl_analysis import SpectrogramEngine
from note_utils import note_to_frequency
from write_mrq import create_same_f0_mrq
from tension_measurement import TensionMeter

if __name__ == "__main__":
    tm = TensionMeter(
        work_dir = Path("X:\\work"), # 建议使用内存盘，因为会产生大量小文件
        note = "A4",
        sample_num = 16384,
        pad_num = 4096,
        sample_rate = 44100
    )
    ampl_range_dict ={"harmonic_0": (-27, -25), #填写谐波振幅的范围，(-27, -25)表示振幅范围为[-27,-25]，步长默认为1.0
                    "harmonic_1": (-25, -27),
                    "harmonic_2": (-26, -29),
                    }
    pprint(tm.generate_ampl_combinations(ampl_range_dict))
    
    file_names = tm.generate_sin_waves(ampl_range_dict)
    #print(file_names)
    tm.process_with_more_sampler(tension_range=(-100, 100), tension_step=20)
    tm.extract_and_store_amplitude()
