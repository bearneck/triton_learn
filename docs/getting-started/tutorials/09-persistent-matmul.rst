持久化矩阵乘法（Persistent Matmul）
=====================================

本脚本展示了使用 Triton 实现矩阵乘法 kernel 的多种持久化方案。
包含朴素（naive）、持久化（persistent）以及基于 TMA（Tensor Memory Accelerator）
等多种 matmul 实现方式。
这些 kernel 支持 FP16 和 FP8 数据类型，但 FP8 实现仅适用于计算能力 >= 9.0 的 CUDA 设备。

本脚本在不同配置下对 Triton 和 cuBLAS 的实现进行基准测试，
并使用 proton 性能分析器进行评估。
用户可通过命令行参数灵活指定矩阵维度和迭代步长。

.. code-block:: bash

    # FP8
    python 09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128

    # FP16
    python 09-persistent-matmul.py --prec fp16 --K_range 128 1024 --K_step 128

注意：当前本教程在共享内存较小的设备（如 RTX-4090）上可能会失败。

.. note::

   `查看源码 <https://github.com/bearneck/triton_learn/blob/master/tutorials/09-persistent-matmul.py>`_
