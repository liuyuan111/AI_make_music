# -*- coding: UTF-* -*-

import os
from music21 import converter, instrument


def print_notes():
    if not os.path.exists("1.mid"):
        raise Exception("MIDI 文件 1.mid 不在此目录下，请添加")

    # 读取 MIDI 文件, 输出 Stream 流类型
    stream = converter.parse("1.mid")

    # 获得所有乐器部分
    parts = instrument.partitionByInstrument(stream)

    if parts:  # 如果有乐器部分，取第一个乐器部分
        notes = parts.parts[0].recurse()
    else:
        notes = stream.flat.notes

    # 打印出每一个元素
    for element in notes:
        print(str(element))


if __name__ == "__main__":
    print_notes()
