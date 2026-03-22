# Triton 官方教程中文翻译

> 本项目将 [Triton](https://triton-lang.org) 官方教程的所有注释与文档字符串翻译为中文，代码逻辑与原版完全一致，可直接运行。

## 什么是 Triton？

[Triton](https://github.com/triton-lang/triton) 是由 OpenAI 开源的 GPU 编程语言与编译器，旨在让开发者能够用类 Python 的语法编写高性能 GPU kernel，无需深入掌握 CUDA PTX 等底层细节，同时达到接近甚至超越 cuBLAS 的性能。

---

## 教程目录

按照官方建议，**建议从第 01 篇开始依次阅读**。

| 文件 | 教程名称 | 主要知识点 |
|------|---------|-----------|
| `tutorials/01-vector-add.py` | **向量加法** | Triton 基本编程模型、`@triton.jit` 装饰器、kernel 验证与基准测试 |
| `tutorials/02-fused-softmax.py` | **融合 Softmax** | kernel 融合、带宽受限算子优化、`tl.reduce` 归约操作 |
| `tutorials/03-matrix-multiplication.py` | **矩阵乘法** | 分块矩阵乘法、多维指针算术、L2 cache 优化、`triton.autotune` 自动调优 |
| `tutorials/04-low-memory-dropout.py` | **低内存 Dropout** | 并行伪随机数生成（Philox 算法）、基于 seed 的确定性 Dropout |
| `tutorials/05-layer-norm.py` | **Layer 归一化** | 前向与反向传播实现、并行归约、VJP 推导 |
| `tutorials/06-fused-attention.py` | **融合注意力机制** | Flash Attention v2、TMA、warp specialization、FP8 支持 |
| `tutorials/07-extern-functions.py` | **外部函数（libdevice）** | 调用 CUDA libdevice / HIP device-lib 数学函数 |
| `tutorials/08-grouped-gemm.py` | **分组 GEMM** | 静态设备端调度、固定 CTA 数量、TMA 加速、多 GEMM 批处理 |
| `tutorials/09-persistent-matmul.py` | **持久化矩阵乘法** | 持久化 kernel、TMA 描述符、warp specialization（Hopper/Blackwell）、FP8 matmul |
| `tutorials/10-block-scaled-matmul.py` | **块缩放矩阵乘法** | FP4/FP8 块缩放格式（nvfp4、mxfp4、mxfp8）、第五代 Tensor Core、AMD CDNA4 |

---

## 环境要求

```bash
# Python 3.8+
pip install triton torch
```

部分高级教程对硬件有额外要求：

- **教程 06、09**：FP8 支持需要 CUDA 计算能力 ≥ 9.0（Hopper 架构，如 H100）
- **教程 10**：块缩放矩阵乘法需要 CUDA 计算能力 = 10（Blackwell 架构，如 B200）或 AMD CDNA4 GPU

---

## 快速开始

```bash
# 运行向量加法教程
python tutorials/01-vector-add.py

# 运行矩阵乘法教程
python tutorials/03-matrix-multiplication.py

# 运行持久化 matmul（FP16）
python tutorials/09-persistent-matmul.py --prec fp16 --K_range 128 1024 --K_step 128

# 运行块缩放 matmul（FP4）
python tutorials/10-block-scaled-matmul.py --format nvfp4
```

---

## 翻译说明

- **注释与文档**：所有 `#` 行注释和 `"""..."""` 文档字符串均已翻译为中文。
- **代码不变**：函数名、变量名、逻辑结构与官方原版完全一致。
- **术语保留**：专业技术术语（`kernel`、`warp`、`block`、`SRAM`、`DRAM`、`TMA`、`GEMM`、`FP8` 等）保留英文，必要时附中文解释。

---

## 原始资源

- 官方文档：https://triton-lang.org/main/index.html
- GitHub 仓库：https://github.com/triton-lang/triton
- 原始教程源码：https://github.com/triton-lang/triton/tree/main/python/tutorials
