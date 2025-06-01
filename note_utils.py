def note_to_frequency(note):
    """
    将类似 "A4" 的音符字符串转换为对应的频率（Hz）。
    参考标准：A4 = 440 Hz
    """
    # 音名到半音的偏移量（相对于 C）
    note_map = {
        'C': -9,
        'C#': -8, 'Db': -8,
        'D': -7,
        'D#': -6, 'Eb': -6,
        'E': -5,
        'F': -4,
        'F#': -3, 'Gb': -3,
        'G': -2,
        'G#': -1, 'Ab': -1,
        'A': 0,
        'A#': 1, 'Bb': 1,
        'B': 2
    }

    if len(note) == 2:
        pitch, octave = note[0], int(note[1])
        offset = note_map.get(pitch, None)
    elif len(note) == 3:
        pitch, octave = note[:2], int(note[2])
        offset = note_map.get(pitch, None)
    else:
        raise ValueError("无效的音符格式")

    if offset is None:
        raise ValueError(f"未知音名: {pitch}")

    # A4 对应 MIDI 编号 69，每半音增加或减少 1
    midi_number = 69 + offset + (octave - 4) * 12

    # 使用公式计算频率：f(n) = 440 * 2^((n - 69)/12)
    frequency = 440.0 * (2 ** ((midi_number - 69) / 12))
    return round(frequency, 2)