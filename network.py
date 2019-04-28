# -*- coding: UTF-8 -*-

"""
RNN-LSTM 循环神经网络
"""

import tensorflow as tf


# 神经网络的模型
def network_model(inputs, num_pitch, weights_file=None):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(
        512,  # LSTM 层神经元的数目是 512，也是 LSTM 层输出的维度
        input_shape=(inputs.shape[1], inputs.shape[2]),  # 输入的形状，对第一个 LSTM 层必须设置
        # return_sequences：控制返回类型
        # - True：返回所有的输出序列
        # - False：返回输出序列的最后一个输出
        # 在堆叠 LSTM 层时必须设置，最后一层 LSTM 可以不用设置
        return_sequences=True  # 返回所有的输出序列（Sequences）
    ))
    model.add(tf.keras.layers.Dropout(0.3))  # 丢弃 30% 神经元，防止过拟合
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512))  # return_sequences 是默认的 False，只返回输出序列的最后一个
    model.add(tf.keras.layers.Dense(256))  # 256 个神经元的全连接层
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_pitch))  # 输出的数目等于所有不重复的音调的数目
    model.add(tf.keras.layers.Activation('softmax'))  # Softmax 激活函数算概率
    # 交叉熵计算误差，使用对 循环神经网络来说比较优秀的 RMSProp 优化器
    # 计算误差（先用 Softmax 计算百分比概率，再用 Cross entropy（交叉熵）来计算百分比概率和对应的独热码之间的误差）
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if weights_file is not None:  # 如果是 生成 音乐时
        # 从 HDF5 文件中加载所有神经网络层的参数（Weights）
        model.load_weights(weights_file)

    return model
