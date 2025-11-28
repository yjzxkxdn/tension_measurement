import numpy as np
import struct
import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}  # 复数需特殊处理
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()  # 数组转为 Python list
        elif obj is np.nan or obj is np.inf or obj is -np.inf:
            return None  # 或者用字符串表示，如 "NaN", "Infinity"
        elif isinstance(obj, np.str_):
            return str(obj)
        elif isinstance(obj, np.bytes_):
            return obj.decode('utf-8')
        elif isinstance(obj, bytes):
            # 将bytes转换为16进制字符串表示
            return obj.hex()
        return super().default(obj)
def read_xchg_file(filename, start=None, end=None):
    """
    读取xchg二进制文件并解析为Python数据结构
    :param filename: 文件名
    :param start: 起始索引 (用于类型7节点)
    :param end: 结束索引 (用于类型7节点)
    :return: 解析后的字典结构
    """
    with open(filename, 'rb') as f:
        # 检查文件头
        header = f.read(4)
        if header == b'data':
            # 直接解析为字典
            return _parse_node(f)
        else:
            print("文件头不是'data'")
            # 读取名称长度和名称
            f.seek(0)
            name_len = f.read(1)[0]
            print(f"文件名长度: {name_len}")
            name = f.read(name_len).decode('ascii')
            print(f"文件名: {name}")
            print("开始解析节点-----")
            # 递归解析节点
            return {name: _parse_node(f, start, end)}

def _parse_node(f, start=None, end=None):
    """递归解析节点为Python数据结构"""
    # 读取节点类型
    node_type = f.read(1)[0]

    if node_type == 1:  # 字典类型
        size = struct.unpack('<i', f.read(4))[0]
        print("----------------------------")
        print("键值对类型节点")
        print(f"键值对数量: {size}")
        print("-------")
        result = {}
        for _ in range(size):
            # 读取键名
            key_len = f.read(1)[0]
            key = f.read(key_len).decode('ascii')
            print(f"键名: {key}")
            # 递归解析值
            result[key] = _parse_node(f, start, end)
        return result
    
    elif node_type == 2:  # 列表类型
        print("列表类型节点")
        dim1 = struct.unpack('<i', f.read(4))[0]
        dim2 = struct.unpack('<i', f.read(4))[0]
        size = dim1 * dim2
        print(f"元素数量: {size}")
        return [_parse_node(f, start, end) for _ in range(size)]
    
    elif node_type == 3:  # 浮点数值
        print("浮点数值节点")
        data = np.frombuffer(f.read(4), dtype=np.float32)[0]
        print(f"元素数值: {data}")
        return data
    
    elif node_type == 5:  # 浮点数组
        print("浮点数组节点")
        dim1 = struct.unpack('<i', f.read(4))[0]
        dim2 = struct.unpack('<i', f.read(4))[0]
        size = dim1 * dim2
        print(f"元素数量: {size}")
        data = np.frombuffer(f.read(size * 4), dtype=np.float32)
        return data.reshape((dim1, dim2))
    
    elif node_type == 6:  # 字节数组
        print("字节数组节点")
        size = struct.unpack('<i', f.read(4))[0]
        print(f"字节数: {size}")
        return f.read(size)
    
    elif node_type == 7:  # 带偏移表的节点数组
        print("带偏移表的节点数组节点")
        count = struct.unpack('<i', f.read(4))[0]
        print(f"节点数量: {count}")
        # 读取偏移表
        offsets = [struct.unpack('<i', f.read(4))[0] for _ in range(count + 1)]
        
        # 计算读取范围
        start_idx = start or 0
        end_idx = end or count
        if end_idx > count:
            end_idx = count
        
        # 保存当前位置
        current_pos = f.tell()
        result = []
        
        # 读取选定范围的节点
        for i in range(start_idx, end_idx):
            f.seek(offsets[i])
            result.append(_parse_node(f, start, end))
        
        # 恢复位置
        f.seek(offsets[count])
        return result
    
    else:
        raise ValueError(f"不支持的节点类型: {node_type}")

# 使用示例
if __name__ == "__main__":

    filename =  r"E:\Program Files\OpenUtau-DiffsingerPack\Singers\神御玥（utau）\niu.wav.llsm"
    # 读取文件
    data = read_xchg_file(filename)
    #print(data)
    # 保存为json文件
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=4)
    
