/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_unaligned_custom_kernel.h"

#if ASCENDC_CPU_DEBUG
#define SET_G_CORE_TYPE_IS_AIC int g_coreType = 1
#else
#define SET_G_CORE_TYPE_IS_AIC
#endif

namespace MatmulCustom {

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulUnalignedKernel<AType, BType, CType, BiasType>::Init(GM_ADDR a,
        GM_ADDR b, GM_ADDR bias, GM_ADDR c, const TCubeTiling& tiling)
{
    this->tiling = tiling;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(bias), tiling.N);


    uint32_t offsetA = 0;
    uint32_t offsetB = 0;
    uint32_t offsetC = 0;
    uint32_t offsetBias = 0;
    CalcOffset(AscendC::GetBlockIdx(), offsetA, offsetB, offsetC, offsetBias);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];

    if(GetSysWorkSpacePtr() == nullptr){
        return;
    }
}

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulUnalignedKernel<AType, BType, CType, BiasType>::Process(AscendC::TPipe* pipe)
{
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    if (tiling.isBias) {
        matmulObj.SetBias(biasGlobal);
    }
    matmulObj.IterateAll(cGlobal);
    matmulObj.End();
}


template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulUnalignedKernel<AType, BType, CType, BiasType>::CalcOffset(
    uint32_t blockIdx, uint32_t& offsetA, uint32_t& offsetB, uint32_t& offsetC, uint32_t& offsetBias)
{
    auto temp0 = AscendC::Ceil(this->tiling.M, this->tiling.singleCoreM);
    auto temp1 = AscendC::Ceil(this->tiling.N, this->tiling.singleCoreN);
    auto temp2 = AscendC::Ceil(this->tiling.Ka, this->tiling.singleCoreK);

    auto divideKCoreNum = this->tiling.usedCoreNum / temp2;

    auto mCoreIndex = (blockIdx % divideKCoreNum) % temp0;
    auto nCoreIndex = (blockIdx % divideKCoreNum) / temp0;
    auto subKIndex = blockIdx / divideKCoreNum;

    offsetA = mCoreIndex * this->tiling.Ka * this->tiling.singleCoreM + subKIndex * this->tiling.singleCoreK;
    offsetB = subKIndex * this->tiling.singleCoreK * this->tiling.N + nCoreIndex * this->tiling.singleCoreN;
    offsetC = mCoreIndex * this->tiling.N * this->tiling.singleCoreM + nCoreIndex * this->tiling.singleCoreN;
    offsetBias = nCoreIndex * this->tiling.singleCoreN;

    uint32_t gmUseM = this->tiling.M - mCoreIndex * this->tiling.singleCoreM;
    uint32_t tailM = gmUseM < this->tiling.singleCoreM ? gmUseM : this->tiling.singleCoreM;
    uint32_t gmUseN = this->tiling.N - nCoreIndex * this->tiling.singleCoreN;
    uint32_t tailN = gmUseN < this->tiling.singleCoreN ? gmUseN : this->tiling.singleCoreN;
    uint32_t gmUseK = this->tiling.Ka - subKIndex * this->tiling.singleCoreK;
    uint32_t tailK = gmUseK < this->tiling.singleCoreK ? gmUseK : this->tiling.singleCoreK;

    if (tailM < this->tiling.singleCoreM || tailN < this->tiling.singleCoreN || tailK < this->tiling.singleCoreK) {
        matmulObj.SetTail(tailM, tailN, tailK);
    }
}
}  // namespace MatmulCustom

namespace {
__aicore__ inline void CopyTiling(TCubeTiling* tiling, GM_ADDR tilingGM)
{
    uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
      *ptr = *(tiling32 + i);
    }
    return;
}
}

extern "C" __global__ __aicore__ void matmul_unaligned_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    if (g_coreType == AscendC::AIV) {
        return;
    }

    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);
    AscendC::TPipe pipe;

    MatmulCustom::MatmulUnalignedKernel<half, half, float, float> matmulUnalignedKernel;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulUnalignedKernel.matmulObj, &tiling);
    matmulUnalignedKernel.Init(a, b, bias, c, tiling);
    matmulUnalignedKernel.Process(&pipe);
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_unaligned_custom_do(uint32_t blockDim, void* stream,
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    matmul_unaligned_custom<<<blockDim, nullptr, stream>>>(a, b, bias, c, workspace, tilingGm);
}
#endif