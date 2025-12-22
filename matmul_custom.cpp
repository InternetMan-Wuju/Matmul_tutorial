/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

//USE : REPLACE:
// tilingMatmul_M tilingMatmul_N tilingMatmul_K tilingMatmul_E 
// tilingScamul_M tilingScamul_N tilingScamul_K tilingScamul_E
//
//
constexpr int32_t  tilingMatmul_M=16; constexpr int32_t tilingMatmul_N=16; constexpr int32_t tilingMatmul_K=16;


__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

class W4A4GroupMatmul {
public:
    __aicore__ inline W4A4GroupMatmul(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, const TCubeTiling &tilingMatmul, AscendC::TPipe *pipe);
    __aicore__ inline void Process(GM_ADDR workspace);
 
    //===================================
    Matmul<MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, int8_t>,
               MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, int8_t>,
               MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, int32_t>> mmMatmul;


    //Global内存，对应你的每一个最初输入和最终输出
    AscendC::GlobalTensor<int8_t> xGlobal;
    AscendC::GlobalTensor<int8_t> wGlobal;
    AscendC::GlobalTensor<int32_t> yGlobal;

    //===========================TWO TILING
    TCubeTiling tilingMatmul;

    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();

    //TQue&BUFFER,对应你需要的输入输出和中间结果的队列,暂存
    //VECIN
    AscendC::TQue<AscendC::TPosition::VECIN, 1> xInQueue_;     
    AscendC::TQue<AscendC::TPosition::VECIN, 1> wInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> yInQueue_;            

    //VECCALC
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCALC_Buf_;     
    AscendC::TBuf<AscendC::TPosition::VECCALC> wCALC_Buf_;       
    AscendC::TBuf<AscendC::TPosition::VECCALC> y_CALC_Buf_;

    //VECOUT
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> xOutQueue_;     
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> wOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yOutQueue_;//

};



__aicore__ inline void W4A4GroupMatmul::Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,const TCubeTiling &tilingMatmul, AscendC::TPipe *pipe)
{
    this->tilingMatmul = tilingMatmul;

    xGlobal.SetGlobalBuffer((__gm__ int8_t *)x, tilingMatmul_M * tilingMatmul_K);
    wGlobal.SetGlobalBuffer((__gm__ int8_t *)w, tilingMatmul_K * tilingMatmul_N);
    yGlobal.SetGlobalBuffer((__gm__ int32_t *)y,  tilingMatmul_M * tilingMatmul_N);
    //====================VECIN
    pipe->InitBuffer(xInQueue_, 1, (tilingMatmul_M * tilingMatmul_K) * sizeof(int8_t)); 
    pipe->InitBuffer(wInQueue_, 1, (tilingMatmul_K * tilingMatmul_N) * sizeof(int8_t));
    pipe->InitBuffer(yInQueue_, 1, (tilingMatmul_M * tilingMatmul_N) * sizeof(int32_t));
    // //====================VECCALC
    pipe->InitBuffer(xCALC_Buf_,tilingMatmul_K*tilingMatmul_M*sizeof(int8_t));
    pipe->InitBuffer(wCALC_Buf_,tilingMatmul_K*tilingMatmul_N*sizeof(int8_t));
    pipe->InitBuffer(y_CALC_Buf_,tilingMatmul_M*tilingMatmul_N*sizeof(int32_t));
    // //====================VECOUT
    pipe->InitBuffer(xOutQueue_, 1, (tilingMatmul_M * tilingMatmul_K) * sizeof(int8_t)); 
    pipe->InitBuffer(wOutQueue_, 1, (tilingMatmul_K * tilingMatmul_N) * sizeof(int8_t));
    pipe->InitBuffer(yOutQueue_, 1, (tilingMatmul_M * tilingMatmul_N) * sizeof(int32_t));

};
__aicore__ inline void W4A4GroupMatmul::CopyIn()
{
      printf("Testblock CopyIN=====================\n");
        AscendC::LocalTensor<int8_t> xLocal = xInQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> wLocal = wInQueue_.AllocTensor<int8_t>();

        AscendC::DataCopy(xLocal, xGlobal, tilingMatmul_M * tilingMatmul_K);
        AscendC::DataCopy(wLocal, wGlobal, tilingMatmul_K * tilingMatmul_N);

        //EnQue->Stage1
        xInQueue_.EnQue(xLocal);
        wInQueue_.EnQue(wLocal);
        printf("Testblock CopyIN====END=================\n");
};
/**
  * @brief  Main process of matmul calculation
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
__aicore__ inline void W4A4GroupMatmul::Process(GM_ADDR workspace)
{
  //CopyIn(GLOBAL->LOCAL,GM->VECIN)
  CopyIn();
  //LOCAL(VECIN) 
  //Deque(LOCALTENSOR)
    AscendC::LocalTensor<int8_t> xLocal = xInQueue_.DeQue<int8_t>();
    AscendC::LocalTensor<int8_t> wLocal = wInQueue_.DeQue<int8_t>();

    AscendC::LocalTensor<int32_t> y_In = yInQueue_.AllocTensor<int32_t>();
    AscendC::LocalTensor<int32_t> y_CALC=y_CALC_Buf_.AllocTensor<int32_t>();
    AscendC::LocalTensor<int32_t> yLocal = yOutQueue_.AllocTensor<int32_t>();
    //FOR RESULT
    

    //======================================IMPORTANT========================
    //https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0102.html
    //GLOBAL->LOCAL / LOCAL->LOCAL /LOCAL->GLOBAL
    //GM->VECIN,VECIN->VECCALC,VECCALC->VECOUT (VECOUT(X,W)->COMPUTE->VECIN(Y) )VECIN->VECCALC->VECOUT->LOCAL->GM
    //以上关系说明了，我们为什么需要进行上下方的搬运，因为不能一步到位
    //=========================================================

    //VECIN->CALC
    AscendC::LocalTensor<int8_t> x_calc = xCALC_Buf_.AllocTensor<int8_t>();
    AscendC::LocalTensor<int8_t> w_calc = wCALC_Buf_.AllocTensor<int8_t>(); 


    AscendC::DataCopy(x_calc,xLocal,tilingMatmul_M*tilingMatmul_K);
    AscendC::DataCopy(w_calc,wLocal,tilingMatmul_K*tilingMatmul_N);

    //CALC->VECOUT

    AscendC::LocalTensor<int8_t> x_out = xOutQueue_.AllocTensor<int8_t>();
    AscendC::LocalTensor<int8_t> w_out = wOutQueue_.AllocTensor<int8_t>(); 


    AscendC::DataCopy(x_out,x_calc,tilingMatmul_M*tilingMatmul_K);
    AscendC::DataCopy(w_out,w_calc,tilingMatmul_K*tilingMatmul_N);


    //=============================COPY OVER============================
    
    //=============================MatMul Cal=================================
    printf("Testblock MatMul Cal Start=====================\n");

    mmMatmul.SetOrgShape(tilingMatmul_M,tilingMatmul_N,tilingMatmul_K);
    mmMatmul.SetTensorA(x_out);
    mmMatmul.SetTensorB(w_out);
    

    //mmMatmul.IterateAll(MatMulIn[sum_line*tilingMatmul_N]);
    while(mmMatmul.Iterate()){ 
      mmMatmul.GetTensorC(y_In);
        };
    
    mmMatmul.End();
    //AscendC::TBuf<AscendC::TPosition::VECCALC> MatMulCALC_Buf_;// 
    //VECIN->VECCALC
    
    AscendC::DataCopy(y_CALC,y_In,tilingMatmul_M*tilingMatmul_N);
    AscendC::DataCopy(yLocal,y_CALC,tilingMatmul_M*tilingMatmul_N);
    printf("First Num: %d\n", yLocal(0));//此处作了一个示例，你可以把yLocal，以及其他的Tensor当作数组，但是要注意不能用[]而是()
    yOutQueue_.EnQue<int32_t>(yLocal);
    printf("Testblock MatMul Cal OVER=====================\n");
    //=============================MatMul Cal OVER=================================     //

    //FREE TENSORS
    xInQueue_.FreeTensor(xLocal);
    wInQueue_.FreeTensor(wLocal);
    yInQueue_.FreeTensor(y_In);
    xCALC_Buf_.FreeTensor(x_calc);
    wCALC_Buf_.FreeTensor(w_calc);
    xOutQueue_.FreeTensor(x_out);
    wOutQueue_.FreeTensor(w_out);
    y_CALC_Buf_.FreeTensor(y_CALC);
    CopyOut();

};

/**
  * @brief  Copy leakyRelu out result to GM.
  * @param  count: Iterate count(once Iterate, compute baseM * baseN).
  * @retval None
  */
__aicore__ inline void W4A4GroupMatmul::CopyOut()
{
    AscendC::LocalTensor<int32_t> yLocal = yOutQueue_.DeQue<int32_t>();
    AscendC::DataCopy(yGlobal, yLocal, tilingMatmul_M*tilingMatmul_N);
    yOutQueue_.FreeTensor(yLocal);

};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR x, GM_ADDR w,  GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGmMatmul)
{
    AscendC::TPipe pipe;
    TCubeTiling tilingMatmul;
    CopyTiling(&tilingMatmul, tilingGmMatmul);
    W4A4GroupMatmul op_kernel;
    op_kernel.Init(x, w, y, tilingMatmul, &pipe);
    // //https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0628.html
    // //↓
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op_kernel.mmMatmul, &op_kernel.tilingMatmul); // Initialize the matmul object.
    op_kernel.Process(workspace);

}