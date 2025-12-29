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
    M = 16 
    K = 16 
    N = 16 
    input_X = np.random.randint(1, 10, [M, K]).astype(np.int8)
    input_W = np.random.randint(1, 10, [K, N]).astype(np.int8)
    # 计算矩阵  
    Y = Matmul(input_X, input_W)   

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
