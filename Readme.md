## 本教程改编自：https://gitee.com/ascend/samples/tree/v1.8-8.3.RC1.alpha002/operator/ascendc/0_introduction/13_matmulleakyrelu_kernellaunch/MatmulLeakyReluInvocation


## 目录结构介绍
```
├── MatMal
│   ├── cmake                               // 编译工程文件--保持默认
│   ├── scripts
│   │   ├── verify_result.py                // 真值对比文件--对比你的算子计算结果和py直接算的结果，一般不需要修改
│   │   └── gen_data.py                     // 输入数据和真值数据生成脚本文件--生成待计算数据和py直接计算的结果，如有需要可以修改
│   ├── CMakeLists.txt                      // 编译工程文件--保持默认，除非你更改下方的 算子.cpp和 算子_tiling.cpp，则需要修改
│   ├── data_utils.h                        // 数据读入写出函数--保持默认
│   ├── main.cpp                            // 主函数，调用算子的应用程序，含CPU域及NPU域调用--如要修改，详情见内部注释
│   ├── matmul_custom_tiling.cpp  // 算子tiling实现--主要编写文件
│   ├── matmul_custom.cpp         // 算子kernel实现--主要编写文件
│   └── run.sh                              // 编译运行算子脚本
```
## 代码实现介绍
本样例中实现的是[m, n, k]固定为[64, 64, 64]的matmul算子。
- kernel实现  
  Matmul算子的数学表达式为：
  ```
  C = A * B
  ```

- 调用实现  
  1. CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；
  2. TODO:NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成。

  应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。

## 运行样例算子

  - 配置环境变量
    --TO BE ADDED

  - 样例执行

    ```bash
    bash run.sh -r [RUN_MODE] -v  [SOC_VERSION]
    bash run.sh -r cpu -v Ascend910B3
    ```
    - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]。
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas 推理系列产品AI Core
      - Atlas A2训练系列产品/Atlas 800I A2推理产品

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
    ```bash
    bash run.sh -r cpu -v Ascendxxxyy
    ```

## 更新说明
| 时间       | 更新事项     | 注意事项                                         |
| ---------- | ------------ | ------------------------------------------------ |
| 2025/12/22 | Release |                                                 |
