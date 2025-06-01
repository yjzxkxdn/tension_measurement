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

def clear_directory_recursive(directory):
    """
    递归地删除目录及其子目录下的所有文件，但保留目录结构。
    """
    directory_path = Path(directory)
    
    if not directory_path.is_dir():
        raise ValueError(f"路径不存在或不是文件夹: {directory}")

    for item in directory_path.rglob('*'):
        if item.is_file():
            item.unlink()
            print(f"已删除文件: {item}")

class TensionMeter:
    def __init__(self, work_dir: Path, note: str = "A4", sample_num=16384, pad_num=4096, sample_rate=44100, ):
        self.work_dir = work_dir
        self.note = note
        self.f0 = note_to_frequency(note)
        
        self.sample_num = sample_num
        self.pad_num = pad_num
        self.total_sample_num = sample_num + pad_num
        self.sample_rate = sample_rate
        
        self.setup_work_directories()
        self.generate_base_mrq()
        
        print(f"work_dir: {self.work_dir}")
        print(f"base_mrq_dir: {self.base_mrq_dir}")
        print(f"input_path: {self.input_path}")
        print(f"more_output_path: {self.more_output_path}")
        
    def setup_work_directories(self):

        self.base_mrq_dir = self.work_dir / "base_mrq_dir"
        self.input_path = self.work_dir / "input_dir" / self.note
        self.more_output_path = self.work_dir / "more_output_dir" / self.note
        
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True, exist_ok=True)
        if not self.base_mrq_dir.exists():
            self.base_mrq_dir.mkdir(parents=True, exist_ok=True)
        if not self.input_path.exists():
            self.input_path.mkdir(parents=True, exist_ok=True)
        if not self.more_output_path.exists():
            self.more_output_path.mkdir(parents=True, exist_ok=True)
            
        clear_directory_recursive(self.base_mrq_dir)
        clear_directory_recursive(self.input_path)
        clear_directory_recursive(self.more_output_path)
        
    def generate_base_mrq(self):
        ampl = np.array([-10], dtype=float)
        sins = generate_sins(self.sample_rate, self.f0, self.total_sample_num, ampl,"db")
        sf.write(str(self.base_mrq_dir / "sin_wave_gen.wav"), sins, self.sample_rate)
        
        result = run_moresampler(str(self.base_mrq_dir / "sin_wave_gen.wav"), str(self.base_mrq_dir / "temp.wav"), self.note, 0)
        print(f"MoreSampler using : \n {result.stdout}")
        
    def generate_ampl_combinations(self, ampl_range_dict: dict, step: float = 1.0):
        """   
        Args:
            ampl_range_dict (dict): 每个键对应一个谐波，值为一个元组表示振幅范围 (start, end)
            step (float): 振幅变化步长，默认1.0

        Returns:
            List[List[float]]: 所有可能的振幅组合
        """
        # 先为每个谐波生成对应的候选振幅列表
        ampl_candidates = []
        for key in ampl_range_dict:
            start, end = ampl_range_dict[key]
            
            # 确保 start <= end，否则交换以避免空数组
            if start > end:
                start, end = end, start
                
            candidates = np.arange(start, end + step/2, step).tolist()
            ampl_candidates.append(candidates)

        # 计算笛卡尔积，得到所有可能的振幅组合
        ampl_combinations = [list(p) for p in product(*ampl_candidates)]

        return ampl_combinations
    
    def generate_sin_waves(self, ampl_range_dict: dict, step: float = 1.0):
        """
        生成不同振幅组合的正弦波音频，并保存为 wav 和对应的参数 json 文件。

        Args:
            ampl_range_dict (dict): 谐波振幅范围字典
            step (float): 振幅变化步长
        """
        # 计算所有可能的振幅组合
        ampl_combinations = self.generate_ampl_combinations(ampl_range_dict, step)

        file_names = []

        # 遍历每个振幅组合，生成对应音频文件
        for i, ampl_combination in tqdm.tqdm(enumerate(ampl_combinations)):
            # 构建谐波参数字符串：a1_a2_a3...
            ampl_str = "_".join(f"{a:.3f}" for a in ampl_combination)
            file_name = f"sin_{i}.wav"
            ampl_json_file_name = f"sin_{i}.json"

            # 生成音频信号
            sins = generate_sins(
                sample_rate=self.sample_rate,
                f0=self.f0,
                sample_num=self.total_sample_num,
                amplitudes=np.array(ampl_combination, dtype=float),
                scale="db"
            )
            
            if np.max(np.abs(sins)) > 1.0:
                raise ValueError(f"音频信号最大值超过 1.0: {np.max(np.abs(sins))}, 振幅组合: {ampl_combination} \n moresampler 会产生错误结果。")

            sf.write(str(self.input_path / file_name), sins, self.sample_rate, subtype='FLOAT')
            file_names.append(file_name)

            with open(str(self.input_path / ampl_json_file_name), "w") as f:
                json.dump({f"ampl_str": ampl_str}, f, indent=2)

        ###
        ### 为了减小f0提取造成的误差，这里根据输入的f0生成一组mrq文件
        ### 
        create_same_f0_mrq(
            input_mrq_path=str(self.base_mrq_dir / "desc.mrq"),
            output_mrq_path=str(self.input_path / "desc.mrq"),
            file_names=file_names,
            base_f0_value=self.f0
        )
        
        return file_names
    
    def process_with_more_sampler(self, tension_range, tension_step):
        file_names = list(self.input_path.glob("*.wav"))
        for file_name in tqdm.tqdm(file_names):
            tension_list = np.arange(tension_range[0], tension_range[1]+tension_step/2, tension_step)
            ################################################################################
            ### More 有一个问题，如果输入音频没有llsm文件，会生成这个文件的llsm               ###
            ### 第二次输入这个文件时会自动使用这个llsm文件，                                 ### 
            ### 但是从llsm文件推理的结果和首次推理的结果不一致，但是差距不大                   ###
            ### 解决方法有两种，第一种是每次都生成新的llsm文件，第二种是都使用缓存的llsm文件。  ###
            ### 我选择了第二种方法，首先推理了一遍，接下来循环时就都使用这个llsm文件了，        ###
            ### 这样就不会造成改变张力时首次推理的和后续的误差。                              ###
            #################################################################################
            run_moresampler(str(self.input_path / file_name), str(self.more_output_path / f"{file_name.stem}_temp.wav"), self.note, 0)
            for tension in tension_list:
                # print(f"Processing {file_name} with tension {tension}")
                output_file_name = file_name.stem + f"_{tension}.wav"
                # output_file_name = f"{file_name.split('.')[0]}_{tension}.wav"
                run_moresampler(str(self.input_path / file_name), str(self.more_output_path / output_file_name), self.note, tension)

    
    def extract_and_store_amplitude(self, ampl_len : int = None):
        """
        提取经过 MoreSampler 处理后的音频文件的频谱振幅，
        并根据原始 ampl_input 分组，按 tension 值记录其对应的 ampl_db_str。
        最终输出一个结构化的 JSON 字符串或文件。
        json文件格式是
        {
                "输入文件的各个谐波振幅str, 由下划线隔开": {
                    "输入的张力值": "输出的频谱振幅str, 由下划线隔开",
                    "张力值2": "振幅str",
                   ...
                    "temp": "首次推理的频谱振幅str"
                }
        }
        
        """
        result = {}
        opt_file_list = list(self.more_output_path.glob("*.wav"))
        for more_opt_file in tqdm.tqdm(opt_file_list):
            # 获取 tension 值
            parts = more_opt_file.stem.split("_")
            if len(parts) < 3:
                raise ValueError(f"文件名格式错误: {more_opt_file.stem}")
            ampl_input_base = parts[1]
            tension = parts[2]

            # 找到原始 ampl_input 对应的 json 文件
            ampl_input_json = self.input_path / f"sin_{ampl_input_base}.json"
            with open(ampl_input_json, "r") as f:
                ampl_input_dict = json.load(f)
            
            ampl_input = ampl_input_dict.get("ampl_str", ampl_input_base)
            
            if ampl_len is None:
                ampl_len = len(ampl_input.split("_"))
            
            signal, sr = sf.read(more_opt_file)
            spec_engine = SpectrogramEngine(
                signal[int(self.pad_num/2): self.sample_num + int(self.pad_num/2)]
            )
            ampl_db = spec_engine.ampl_analysis_czt(self.f0, sr, ampl_len, "db")
            ampl_db_str = "_".join(f"{a:.3f}" for a in ampl_db)

            if ampl_input not in result:
                result[ampl_input] = {}
            result[ampl_input][tension] = ampl_db_str

        output_json_path = self.work_dir / f"{self.note}_ampl_tension_mapping.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result


if __name__ == "__main__":
    tm = TensionMeter(
        work_dir = Path("X:\\work"), # 建议使用内存盘，因为会产生大量小文件
        note = "A4",
        sample_num = 16384,
        pad_num = 4096,
        sample_rate = 44100
    )
    ampl_range_dict ={"harmonic_0": (-30, -25), #填写谐波振幅的范围，(-30, -25)表示振幅范围为[-30,-25]，步长默认为1.0
                    "harmonic_1": (-26, -27),
                    "harmonic_2": (-27, -29),
                    }
    pprint(tm.generate_ampl_combinations(ampl_range_dict))
    
    file_names = tm.generate_sin_waves(ampl_range_dict)
    #print(file_names)
    tm.process_with_more_sampler(tension_range=(-100, 100), tension_step=2)
    tm.extract_and_store_amplitude()