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


constexpr int32_t CORE_NUM = 8;//假设有8核

//Copytiling函数,自带，未作任何改动
__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}
//算子类
class Matmul_custom {
public:
    __aicore__ inline Matmul_custom(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, const TCubeTiling &tilingMatmul, AscendC::TPipe *pipe,int core_num);
    __aicore__ inline void Process(GM_ADDR workspace);
    // __aicore__ inline void CalcGMOffset(int blockIdx, const TCubeTiling &tiling, int &offsetA, int &offsetB, int &offsetC,
    //                                 int &tailM, int &tailN, bool isTransA, bool isTransB)
    //===================================
    //在此处定义你的Matmul对象，和tiling.cpp中编写的保持一致
    Matmul<MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, int8_t>,
               MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, int8_t>,
               MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, int32_t>> mmMatmul;


    //Global内存，对应你的每一个最初输入和最终输出
    AscendC::GlobalTensor<int8_t> xGlobal;
    AscendC::GlobalTensor<int8_t> wGlobal;
    AscendC::GlobalTensor<int32_t> yGlobal;

    TCubeTiling tilingMatmul;
    //有关TCubeTiling的说明:
    //https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0673.html#ZH-CN_TOPIC_0000002446556620__table1563162142915
    int core_num;//核计数
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    //TQue&BUFFER,对应你需要的输入输出和中间结果的队列,属于上文中提到的计算暂存部分
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


//初始化计算开始的时候要用的暂存队列和Buffer,以及Global
__aicore__ inline void Matmul_custom::Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,const TCubeTiling &tilingMatmul, AscendC::TPipe *pipe,int core_num)
{
    this->tilingMatmul = tilingMatmul;
    this->core_num = core_num;
    xGlobal.SetGlobalBuffer((__gm__ int8_t *)x, tilingMatmul.M * tilingMatmul.Ka);//Ka代表左矩阵列数，Kb代表右矩阵行数
    wGlobal.SetGlobalBuffer((__gm__ int8_t *)w, tilingMatmul.Ka * tilingMatmul.N*CORE_NUM);//注意Global的大小是整体的
    yGlobal.SetGlobalBuffer((__gm__ int32_t *)y,  tilingMatmul.M * tilingMatmul.N*CORE_NUM);
    //====================VECIN
    pipe->InitBuffer(xInQueue_, 1, (tilingMatmul.M * tilingMatmul.Ka) * sizeof(int8_t)); 
    pipe->InitBuffer(wInQueue_, 1, (tilingMatmul.Ka * tilingMatmul.N) * sizeof(int8_t));
    pipe->InitBuffer(yInQueue_, 1, (tilingMatmul.M * tilingMatmul.N) * sizeof(int32_t));
    // //====================VECCALC
    pipe->InitBuffer(xCALC_Buf_,tilingMatmul.Ka*tilingMatmul.M*sizeof(int8_t));
    pipe->InitBuffer(wCALC_Buf_,tilingMatmul.Ka*tilingMatmul.N*sizeof(int8_t));
    pipe->InitBuffer(y_CALC_Buf_,tilingMatmul.M*tilingMatmul.N*sizeof(int32_t));
    // //====================VECOUT
    pipe->InitBuffer(xOutQueue_, 1, (tilingMatmul.M * tilingMatmul.Ka) * sizeof(int8_t)); 
    pipe->InitBuffer(wOutQueue_, 1, (tilingMatmul.Ka * tilingMatmul.N) * sizeof(int8_t));
    pipe->InitBuffer(yOutQueue_, 1, (tilingMatmul.M * tilingMatmul.N) * sizeof(int32_t));

};
__aicore__ inline void Matmul_custom::CopyIn()
{
      //此处有关SliceInfo的编写，请参考https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0105.html 图一和其下方的解释
      //此处作切片搬运，因为我们将运算输入切片到每个核上
      AscendC::SliceInfo srcSliceInfoW[2];
      AscendC::SliceInfo dstSliceInfoW[2];
      AscendC::SliceInfo srcSliceInfoInW[2] = {{core_num*tilingMatmul.N, core_num*tilingMatmul.N+tilingMatmul.N-1, 0, 1, tilingMatmul.N*CORE_NUM}, {0, tilingMatmul.Ka-1, 0, 1, tilingMatmul.Ka}};
      AscendC::SliceInfo dstSliceInfoInW[2] = {{0, tilingMatmul.N-1, 0, 1, tilingMatmul.N}, {0, tilingMatmul.Ka-1, 0, 1, tilingMatmul.Ka}};
      for(int i=0;i<2;i++)
      {
          srcSliceInfoW[i].startIndex=srcSliceInfoInW[i].startIndex;
          srcSliceInfoW[i].endIndex=srcSliceInfoInW[i].endIndex;
          srcSliceInfoW[i].stride=srcSliceInfoInW[i].stride;
          srcSliceInfoW[i].burstLen=srcSliceInfoInW[i].burstLen;
          srcSliceInfoW[i].shapeValue=srcSliceInfoInW[i].shapeValue;

          dstSliceInfoW[i].startIndex=dstSliceInfoInW[i].startIndex;
          dstSliceInfoW[i].endIndex=dstSliceInfoInW[i].endIndex;
          dstSliceInfoW[i].stride=dstSliceInfoInW[i].stride;
          dstSliceInfoW[i].burstLen=dstSliceInfoInW[i].burstLen;
          dstSliceInfoW[i].shapeValue=dstSliceInfoInW[i].shapeValue;

      }

      //
      //printf("Testblock CopyIN=====================\n");
        AscendC::LocalTensor<int8_t> xLocal = xInQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> wLocal = wInQueue_.AllocTensor<int8_t>();
      
        //DataCopy:(目的，源，大小)
        AscendC::DataCopy(xLocal, xGlobal, tilingMatmul.M * tilingMatmul.Ka);
        AscendC::DataCopy(wLocal, wGlobal, dstSliceInfoW, srcSliceInfoW, 2);//w_scale��Ƭ���ݰ���

        //别忘了入队
        xInQueue_.EnQue(xLocal);
        wInQueue_.EnQue(wLocal);
        //Test Block=====
        // if(core_num==0||core_num==1){
        // printf("W test Block ,corenum:%d Brick:\n",core_num);
        // for(int i=0;i<32*32;i++){
        //   if(i%32==0) printf("\n");
        //   printf("%d ",(int)wLocal(i));
        // }
        // printf("\n");
        // }
        //Test Block=====
        //printf("Testblock CopyIN====END=================\n");
        //若把W看作一个8个左右链接的32x32矩阵的话，则此时我们就搬运了一个矩阵，后面和单核一样了
};
/**
  * @brief  Main process of matmul calculation
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
__aicore__ inline void Matmul_custom::Process(GM_ADDR workspace)
{
  //CopyIn(GLOBAL->LOCAL,GM->VECIN)
  CopyIn();
  //LOCAL(VECIN) 
  //中间搬运流程
  //Deque(LOCALTENSOR)
    AscendC::LocalTensor<int8_t> xLocal = xInQueue_.DeQue<int8_t>();
    AscendC::LocalTensor<int8_t> wLocal = wInQueue_.DeQue<int8_t>();

    AscendC::LocalTensor<int32_t> y_In = yInQueue_.AllocTensor<int32_t>();
    AscendC::LocalTensor<int32_t> y_CALC=y_CALC_Buf_.AllocTensor<int32_t>();
    AscendC::LocalTensor<int32_t> yLocal = yOutQueue_.AllocTensor<int32_t>();
    //FOR RESULT
    



    //VECIN->CALC
    AscendC::LocalTensor<int8_t> x_calc = xCALC_Buf_.AllocTensor<int8_t>();
    AscendC::LocalTensor<int8_t> w_calc = wCALC_Buf_.AllocTensor<int8_t>(); 


    AscendC::DataCopy(x_calc,xLocal,tilingMatmul.M*tilingMatmul.Ka);
    AscendC::DataCopy(w_calc,wLocal,tilingMatmul.Ka*tilingMatmul.N);

    //CALC->VECOUT

    AscendC::LocalTensor<int8_t> x_out = xOutQueue_.AllocTensor<int8_t>();
    AscendC::LocalTensor<int8_t> w_out = wOutQueue_.AllocTensor<int8_t>(); 


    AscendC::DataCopy(x_out,x_calc,tilingMatmul.M*tilingMatmul.Ka);
    AscendC::DataCopy(w_out,w_calc,tilingMatmul.Ka*tilingMatmul.N);


    //=============================COPY OVER============================

    //=============================MatMul Cal=================================
    //printf("Testblock MatMul Cal Start=====================\n");

    mmMatmul.SetOrgShape(tilingMatmul.M,tilingMatmul.N,tilingMatmul.Ka);
    mmMatmul.SetTensorA(x_out);
    mmMatmul.SetTensorB(w_out);
    

    //mmMatmul.IterateAll(MatMulIn[sum_line*tilingMatmul.N]);
    while(mmMatmul.Iterate()){ 
      mmMatmul.GetTensorC(y_In);
        };
    
    mmMatmul.End();
    //AscendC::TBuf<AscendC::TPosition::VECCALC> MatMulCALC_Buf_;// 
    //VECIN->VECCALC
    
    AscendC::DataCopy(y_CALC,y_In,tilingMatmul.M*tilingMatmul.N);
    AscendC::DataCopy(yLocal,y_CALC,tilingMatmul.M*tilingMatmul.N);
    //printf("First Num: %d\n", yLocal(0));//此处作了一个示例，你可以把yLocal，以及其他的Tensor当作数组，但是要注意不能用[]而是()，[]代表的是取到这个长度的Tensor
    //Test Block=====
    // if(core_num==0){
    // printf("Result test Block IN Local(Currect) ,corenum:%d Brick:\n",core_num);
    // for(int i=0;i<32*32;i++){
    //   if(i%32==0) printf("\n");
    //   printf("%d ",(int)yLocal(i));
    // }
    // printf("\n");
    // }
    //Test Block=====
    yOutQueue_.EnQue<int32_t>(yLocal);
    //printf("Testblock MatMul Cal OVER=====================\n");
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
//搬出数据到最终所需的地方，也就是（Result）GLOBAL
__aicore__ inline void Matmul_custom::CopyOut()
{

  AscendC::LocalTensor<int32_t> yLocal = yOutQueue_.DeQue<int32_t>();
  //注意CopyOut的修改,此处作普通搬运
  //https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0265.html
  //表五，以及VECIN/VECOUT->GM相关的介绍
  DataCopyParams CopyParams;
  CopyParams.blockCount = tilingMatmul.Ka;
  CopyParams.blockLen = tilingMatmul.N*(sizeof(int32_t))/32;
  //也可以直接1块，然后长度是Ka*N*sizeof(int32_t)/8
  CopyParams.srcStride = 0;
  CopyParams.dstStride = tilingMatmul.N*(sizeof(int32_t))/32 * (CORE_NUM-1);
  AscendC::DataCopy(yGlobal[core_num*tilingMatmul.N], yLocal, CopyParams);//此处yGlobal[core_num*tilingMatmul.N],意思是截取从core_num*N之后的元素组成的向量
  //Test Block=====
  // if(core_num==0){
  // printf("Result test Block ,corenum:%d Brick:\n",core_num);
  // for(int i=0;i<32;i++){
  //   for(int j=0;j<32;j++){
  //     printf("%d ",(int)yGlobal(i*32*8+j));
  // }
  // printf("\n");
  // }
  // }
  //Test Block=====
  //AscendC::DataCopy(yGlobal, yLocal, tilingMatmul.M*tilingMatmul.N);
  yOutQueue_.FreeTensor(yLocal);



};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR x, GM_ADDR w,  GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGmMatmul)
{
    int core_num = AscendC::GetBlockIdx();
    if(core_num >= CORE_NUM)
    {
        core_num=CORE_NUM-1;
    }
    //以上代码，我们假设有8核，而多出来的核全部作第七（从0开始）核一样的运算
    AscendC::TPipe pipe;
    TCubeTiling tilingMatmul;
    CopyTiling(&tilingMatmul, tilingGmMatmul);
    Matmul_custom op_kernel;
    op_kernel.Init(x, w, y, tilingMatmul, &pipe,core_num);
    // //https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0628.html
    // //↓注意，尽管文档中说可以注册多个算子对象，但是最好还是只用一个，否则可能导致不可预知的错误
    //注册多个示例：    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op_kernel.mmMatmul, &op_kernel.tilingMatmul,op2_kernel.mmCal, &op2_kernel.tilingCal);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op_kernel.mmMatmul, &op_kernel.tilingMatmul); // Initialize the matmul object.
    op_kernel.Process(workspace);

}