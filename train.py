# -*- coding: UTF-8 -*-

"""
训练神经网络，将参数（Weight）存入 HDF5 文件
"""

import numpy as np
import tensorflow as tf

from utils import *
from network import *

"""
==== 一些术语的概念 ====
# Batch size : 批次(样本)数目。一次迭代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目。Batch size 越大，所需的内存就越大
# Iteration : 迭代。每一次迭代更新一次权重（网络参数），每一次权重更新需要 Batch size 个数据进行 Forward 运算，再进行 BP 运算
# Epoch : 纪元/时代。所有的训练样本完成一次迭代

# 假如 : 训练集有 1000 个样本，Batch_size=10
# 那么 : 训练完整个样本集需要： 100 次 Iteration，1 个 Epoch
# 但一般我们都不止训练一个 Epoch
"""


# 训练神经网络
def train():
    notes = get_notes()

    # 得到所有不重复（因为用了 set）的音调数目
    num_pitch = len(set(notes))

    network_input, network_output = prepare_sequences(notes, num_pitch)

    model = network_model(network_input, num_pitch)

    filepath = "weights-{epoch:02d}-{loss:.4f}.hdf5"

    # 用 Checkpoint（检查点）文件在每一个 Epoch 结束时保存模型的参数（Weights）
    # 不怕训练过程中丢失模型参数。可以在我们对 Loss（损失）满意了的时候随时停止训练
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,  # 保存的文件路径
        monitor='loss',  # 监控的对象是 损失（loss）
        verbose=0,
        save_best_only=True,  # 不替换最近的数值最佳的监控对象的文件
        mode='min'      # 取损失最小的
    )
    callbacks_list = [checkpoint]

    # 用 fit 方法来训练模型
    model.fit(network_input, network_output, epochs=100, batch_size=64, callbacks=callbacks_list)


def prepare_sequences(notes, num_pitch):
    """
    为神经网络准备好供训练的序列
    """
    sequence_length = 100  # 序列长度

    # 得到所有音调的名字
    pitch_names = sorted(set(item for item in notes))

    # 创建一个字典，用于映射 音调 和 整数
    pitch_to_int = dict((pitch, num) for num, pitch in enumerate(pitch_names))

    # 创建神经网络的输入序列和输出序列
    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]

        network_input.append([pitch_to_int[char] for char in sequence_in])
        network_output.append(pitch_to_int[sequence_out])

    n_patterns = len(network_input)

    # 将输入的形状转换成神经网络模型可以接受的
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # 将 输入 标准化 / 归一化
    # 归一话可以让之后的优化器（optimizer）更快更好地找到误差最小值
    network_input = network_input / float(num_pitch)

    # 将期望输出转换成 {0, 1} 组成的布尔矩阵，为了配合 categorical_crossentropy 误差算法使用
    network_output = tf.keras.utils.to_categorical(network_output)

    return network_input, network_output


if __name__ == '__main__':
    train()
