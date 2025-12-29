/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_matmul_custom.h"
#else
#include "tikicpulib.h"
extern "C" void matmul_custom(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t * );//注意按照.cpp填写名称和参数个数
#endif
extern void GenerateTilingMatmul(const char *socVersion, uint8_t *tilingBuf);

int32_t main(int32_t argc, char *argv[])
{
    const char *socVersion = SOC_VERSION;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);


    //定义矩阵大小，注意各个文件保持一致
    int32_t M = 32;
    int32_t K = 32;
    int32_t N = 32*8;//假设有8核,每核计算32列

    //
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    //==============此处定义了你在Scripts中生成的输入文件大小。以及输出文件大小
    size_t input_X_Size = M * K * sizeof(int8_t);
    size_t input_W_Size = K * N * sizeof(int8_t);
    size_t y_Size = M * N * sizeof(int32_t);
    //==============

    size_t tilingFileSize = sizeof(TCubeTiling);

    uint8_t *tilingBufMatmul = (uint8_t *)malloc(tilingFileSize);


    GenerateTilingMatmul(socVersion, tilingBufMatmul);

    //若要使用多个Matmul对象，注意修改上方的代码.还有文件最末尾的释放不要忘了，不然会报错。
    // uint8_t *tilingBufMatmul2 = (uint8_t *)malloc(tilingFileSize);
    // GenerateTilingMatmul2(socVersion, tilingBufMatmul2);

#ifdef CUSTOM_ASCEND310P//NOTUSE
    uint32_t blockDim = 2;
#else
    uint32_t blockDim = 8;
    //uint32_t blockDim = 8;//假设有8核
#endif

//uint32_t blockDim = 1;

#ifdef ASCENDC_CPU_DEBUG//=============USE
    //=============
        //printf("TestBlock1\n");
    //分配输入内存
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(input_X_Size);
    uint8_t *w = (uint8_t *)AscendC::GmAlloc(input_W_Size);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(y_Size);
    //=============

    uint8_t *tilingMatmul = (uint8_t *)AscendC::GmAlloc(tilingFileSize);
    ///
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    //=============?

    ReadFile("./input/x.bin", input_X_Size, x, input_X_Size);
    ReadFile("./input/w.bin", input_W_Size, w, input_W_Size);


    memcpy_s(tilingMatmul, tilingFileSize, tilingBufMatmul, tilingFileSize);
    //运行算子↓
    ICPU_RUN_KF(matmul_custom, blockDim, x, w, y, workspace, tilingMatmul);
    //运行算子↑
    WriteFile("./output/output.bin", y, y_Size);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)w);

    AscendC::GmFree((void *)y);


    AscendC::GmFree((void *)tilingMatmul);

    AscendC::GmFree((void *)workspace);

#else//==================================================================================================?????????
//This must for NO:CUSTOM_ASCEND310P&&NO:ASCENDC_CPU_DEBUG
//NOT   USED
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inputAHost;
    uint8_t *inputADevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputAHost), aFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", aFileSize, inputAHost, aFileSize);
    CHECK_ACL(aclrtMemcpy(inputADevice, aFileSize, inputAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputBHost;
    uint8_t *inputBDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBHost), bFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", bFileSize, inputBHost, bFileSize);
    CHECK_ACL(aclrtMemcpy(inputBDevice, bFileSize, inputBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *outputCHost;
    uint8_t *outputCDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&outputCHost), cFileSize));
    CHECK_ACL(aclrtMalloc((void **)&outputCDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *inputBiasHost;
    uint8_t *inputBiasDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBiasHost), biasFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBiasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/bias.bin", biasFileSize, inputBiasHost, biasFileSize);
    CHECK_ACL(aclrtMemcpy(inputBiasDevice, biasFileSize, inputBiasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *tilingHost;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(matmul_leakyrelu_custom)
    (blockDim, stream, inputADevice, inputBDevice, inputBiasDevice, outputCDevice, workspaceDevice, tilingDevice);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtFree(inputADevice));
    CHECK_ACL(aclrtFreeHost(inputAHost));
    CHECK_ACL(aclrtFree(inputBDevice));
    CHECK_ACL(aclrtFreeHost(inputBHost));
    CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", outputCHost, cFileSize);
    CHECK_ACL(aclrtFree(outputCDevice));
    CHECK_ACL(aclrtFreeHost(outputCHost));
    CHECK_ACL(aclrtFree(inputBiasDevice));
    CHECK_ACL(aclrtFreeHost(inputBiasHost));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(workspaceDevice));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBufMatmul);
    return 0;
}