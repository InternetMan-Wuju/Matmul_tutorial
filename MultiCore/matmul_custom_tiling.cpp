/**
 * @file matmul_leakyrelu_custom_tiling.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
using namespace std;

/**
  * @brief  Generate matmul tiling.
  * @param  socVersion: Platform socversion.
  * @param  tilingBuf data buffer.
  */
void GenerateTilingMatmul(const char *socVersion, uint8_t *tilingBuf)
{

    constexpr int32_t M = 32;
    constexpr int32_t K = 32;
    constexpr int32_t N = 32;//这里仍然是单核的运算大小。

    //左矩阵定义
    TPosition XPosition = TPosition::VECOUT;
    CubeFormat XFormat = CubeFormat::ND;
    DataType XDtype = DataType::DT_INT8;
    bool isTransX = false;

    //右矩阵定义
    TPosition WPosition = TPosition::VECOUT;
    CubeFormat WFormat = CubeFormat::ND;
    DataType WDtype = DataType::DT_INT8;
    bool isTransW = false;

    //结果矩阵定义
    TPosition MatmulPosition = TPosition::VECIN;
    CubeFormat MatmulFormat = CubeFormat::ND;
    DataType MatmulDtype = DataType::DT_INT32;
    //此处可能有疑问，int8*int8,为什么不用16位？因为不支持，编译器会报错。报错信息中，支持的数据的列表就没有INT16。

    //我们没有bias
    bool isBias = false;


    //以下未作太多修改，按自己的定义编写即可
    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);//注意这里为MultiCoreMatmulTiling
    tilingApi.SetDim(8); //假设有8核
    tilingApi.SetAType(XPosition, XFormat, XDtype, isTransX);
    tilingApi.SetBType(WPosition, WFormat, WDtype, isTransW);
    tilingApi.SetCType(MatmulPosition, MatmulFormat, MatmulDtype);


    tilingApi.SetOrgShape(M, N, K);  
    tilingApi.SetShape(M, N, K); 
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t step1Mul = tilingApi.GetTiling(tilingData);

    if (step1Mul == -1) {
        std::cout << "gen tiling failed, this code is from tiling.cpp" << std::endl;
    }
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);

    return;
}
