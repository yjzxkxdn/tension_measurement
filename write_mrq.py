
import os
import struct
import binascii
import json
from pprint import pprint


class MrqFile:
    def __init__(self, file_path: str = None):
        self.header = {
            "magic": b"mrq\x00",
            "version": 1,
            "num_entries": 0,
        }
        self.entries = []

        if file_path:
            self.read_from_file(file_path)

    # ================== 文件读写 ==================
    def read_from_file(self, file_path: str):
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        self.parse(binary_data)

    def write_to_file(self, file_path: str):
        with open(file_path, 'wb') as f:
            f.write(self.build())

    # ================== 解析与构建 ==================
    def parse(self, data: bytes):
        offset = 0

        self.header["magic"] = data[offset:offset + 4]
        offset += 4

        self.header["version"] = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4

        num_entries = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4

        entries = []
        for _ in range(num_entries):
            filename_len = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4

            filename_bytes = data[offset:offset + filename_len * 2]
            filename = filename_bytes.decode('utf-16-le').rstrip('\x00')
            offset += filename_len * 2

            unknown1 = data[offset:offset + 4]
            offset += 4

            num_f0s = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4

            sample_rate = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4

            unknown2 = data[offset:offset + 4]
            offset += 4

            f0s = []
            for _ in range(num_f0s):
                f0 = struct.unpack('<f', data[offset:offset + 4])[0]
                f0s.append(f0)
                offset += 4

            unknown3 = data[offset:offset + 8]
            offset += 8

            entry = {
                'name': filename,
                'sample_rate': sample_rate,
                'base_f0_list': f0s,
                'unknown1': binascii.hexlify(unknown1).decode('utf-8').upper(),
                'unknown2': binascii.hexlify(unknown2).decode('utf-8').upper(),
                'unknown3': binascii.hexlify(unknown3).decode('utf-8').upper(),
            }
            entries.append(entry)

        self.entries = entries
        self.header["num_entries"] = num_entries

    def build(self) -> bytes:
        data = bytearray()

        data.extend(self.header["magic"])
        data.extend(struct.pack('<I', self.header["version"]))

        num_entries_pos = len(data)
        data.extend(b'\x00\x00\x00\x00')  # 占位符

        entries_binary = []

        for entry in self.entries:
            entry_data = bytearray()

            name = entry['name']
            sample_rate = entry['sample_rate']
            base_f0_list = entry.get('base_f0_list', [])
            unknown1 = binascii.unhexlify(entry['unknown1'])
            unknown2 = binascii.unhexlify(entry['unknown2'])
            unknown3 = binascii.unhexlify(entry['unknown3'])

            num_f0s = len(base_f0_list)

            filename_utf16 = name.encode('utf-16-le') + b'\x00\x00'
            filename_len = len(filename_utf16) // 2

            entry_data.extend(struct.pack('<I', filename_len))
            entry_data.extend(filename_utf16)

            entry_data.extend(unknown1)
            entry_data.extend(struct.pack('<I', num_f0s))
            entry_data.extend(struct.pack('<I', sample_rate))
            entry_data.extend(unknown2)

            for f0 in base_f0_list:
                entry_data.extend(struct.pack('<f', f0))

            entry_data.extend(unknown3)

            entries_binary.append(entry_data)

        for ed in entries_binary:
            data.extend(ed)

        num_entries = len(self.entries)
        struct.pack_into('<I', data, num_entries_pos, num_entries)

        return bytes(data)

    # ================== Entry 管理 ==================

    def get_entry_names(self):
        """获取所有条目名称"""
        return [entry['name'] for entry in self.entries]

    def find_entry(self, name: str):
        """通过名字查找条目"""
        for entry in self.entries:
            if entry['name'] == name:
                return entry
        return None

    def add_entry(self, entry_dict: dict):
        """添加一个新条目"""
        required_fields = ['name', 'sample_rate', 'base_f0_list',
                           'unknown1', 'unknown2', 'unknown3']
        if not all(k in entry_dict for k in required_fields):
            raise ValueError("缺少必要字段")
        self.entries.append(entry_dict)
        self.header["num_entries"] = len(self.entries)

    def remove_entry(self, name: str):
        """删除指定名称的条目"""
        original_count = len(self.entries)
        self.entries = [e for e in self.entries if e['name'] != name]
        self.header["num_entries"] = len(self.entries)
        return original_count != len(self.entries)

    def update_entry(self, name: str, update_dict: dict):
        """更新指定名称的条目字段"""
        entry = self.find_entry(name)
        if entry is None:
            return False
        entry.update(update_dict)
        return True

    # ================== 工具方法 ==================

    def print_info(self):
        """打印解析结果"""
        print("解析结果:")
        pprint({
            "Magic": self.header["magic"],
            "Version": self.header["version"],
            "Entries": self.entries
        })

    def print_summary(self):
        """打印简要摘要信息"""
        print(f"Magic: {self.header['magic']}")
        print(f"Version: {self.header['version']}")
        print(f"Total Entries: {len(self.entries)}")
        for i, entry in enumerate(self.entries):
            print(f"{i+1}. Name: {entry['name']}")

    def to_json(self, file_path: str = None) -> dict:
        """将 MRQ 数据转换为 JSON 可序列化结构"""
        result = {
            "header": self.header,
            "entries": self.entries
        }

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    @classmethod
    def from_json(cls, json_data: dict):
        """从 JSON 构建 MRQ 实例"""
        instance = cls()
        instance.header = json_data["header"]
        instance.entries = json_data["entries"]
        return instance
    
def create_same_f0_mrq(
    input_mrq_path: str,
    output_mrq_path: str,
    file_names: list[str],
    base_f0_value: float
):
    # 读取原始 MRQ 文件
    mrq_template = MrqFile(input_mrq_path)

    if not mrq_template.entries:
        raise ValueError("输入 MRQ 文件中没有条目")

    # 取第一个 entry 的 f0 列表长度作为模板
    template_entry = mrq_template.entries[0]
    f0_length = len(template_entry['base_f0_list'])
    
    sample_rate = template_entry['sample_rate']
    unknown1 = template_entry['unknown1']
    unknown2 = template_entry['unknown2']
    unknown3 = template_entry['unknown3']

    # 构建新的 entries
    new_entries = []

    for name in file_names:
        new_entry = {
            'name': name,
            'sample_rate': sample_rate,
            'base_f0_list': [base_f0_value] * f0_length,
            'unknown1': unknown1,
            'unknown2': unknown2,
            'unknown3': unknown3
        }
        new_entries.append(new_entry)

    # 替换 entries
    mrq_template.entries = new_entries
    # 更新 num_entries
    mrq_template.header["num_entries"] = len(new_entries)

    # 写出新文件
    mrq_template.write_to_file(output_mrq_path)
    print(f"成功生成新 MRQ 文件：{output_mrq_path}")

if __name__ == '__main__':
    mrq = MrqFile('./sin_wave/desc.mrq')
    mrq.print_info()

    # 构建并保存为新文件
    output_path = 'desc_new.mrq'
    mrq.write_to_file(output_path)

    # 重新读取验证一致性
    mrq_reloaded = MrqFile(output_path)
    mrq_reloaded.print_info()

    # 打印比较结果
    print("原始数据与重建数据是否一致？")
    print(mrq.entries == mrq_reloaded.entries)
    
    
    print("测试创建新 MRQ 文件")
    
    input_mrq = "./sin_wave/desc.mrq"
    output_mrq = "./sin_wave/new_desc.mrq"

    file_names = [
        "singer_a_01",
        "singer_a_02",
        "singer_a_03",
        "singer_b_01",
        "singer_b_02"
    ]

    base_f0_value = 888
    
    create_same_f0_mrq(
        input_mrq_path=input_mrq,
        output_mrq_path=output_mrq,
        file_names=file_names,
        base_f0_value=base_f0_value
    )
    
    # 验证新文件是否正确生成
    mrq_new = MrqFile(output_mrq)
    mrq_new.print_info()