#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import os

def gen_test_data():  
    M = 32 +1
    K = 32 +1
    N = 32*8 +1
    input_X = np.random.randint(1, 4, [M, K]).astype(np.int8)
    input_W = np.random.randint(1, 4, [K, N]).astype(np.int8)

    # first_row = input_W[0, :]
    # print("W 的第一行（shape = {}）：".format(first_row.shape))
    # print(first_row)
    # first_column = input_W[:, 0]
    # print("W 的第一列（shape = {}）：".format(first_column.shape))
    # print(first_column)
    #测试用第一分块
    # first_block = input_W[0:32, 0:32]
    # print("W 的第一分块（shape = {}）：".format(first_block.shape))
    # print(first_block)
    # second_block = input_W[0:32, 32:64]
    # print("W 的第二分块（shape = {}）：".format(second_block.shape))
    # print(second_block)

    # 计算矩阵  
    Y = Matmul(input_X, input_W)   
    #测试用第一分块
    # first_block = Y[0:32, 0:32]
    # print("Y 的第一分块（shape = {}）：".format(first_block.shape))
    # print(first_block)
    # second_block = Y[0:32, 32:64]
    # print("Y 的第二分块（shape = {}）：".format(second_block.shape))
    # print(second_block)
    # 创建输出文件夹  
    os.makedirs("input", exist_ok=True)  
    os.makedirs("output", exist_ok=True)  

    # 保存
    input_X.tofile("./input/x.bin")  
    input_W.tofile("./input/w.bin")  

    Y.tofile("./output/golden.bin")  
    # 使用 np.savetxt 将转换后的数组保存为 TXT 文件  
    np.savetxt("./output/golden.txt", Y, fmt='%d')  # fmt='%.4f' 指定保留四位小数  

def Matmul(X, W):
    M, K = X.shape
    K_w, N = W.shape
    assert K == K_w, "X和W的K维度必须相等"

    matmul = np.dot(X.astype(np.int32), W.astype(np.int32))
    
    return matmul

if __name__ == "__main__":
    gen_test_data()
