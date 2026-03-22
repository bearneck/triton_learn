triton.language.extra.cuda
==========================

.. currentmodule:: triton.language.extra.cuda

``triton.language.extra.cuda`` 提供了 CUDA 特有的扩展功能。

程序化依赖启动
--------------

程序化依赖启动（Programmatic Dependent Launch，PDL）允许 GPU kernel 在完成部分工作后立即触发下一个 kernel，
而无需返回 CPU，从而显著减少多 kernel 流水线中的启动延迟。需要 CUDA 计算能力 ≥ 9.0（Hopper 及以上）。

.. py:function:: gdc_wait()

   在 kernel 内部等待，直到前驱 kernel 通过 ``gdc_launch_dependents`` 发出信号后才继续执行。
   用于实现 kernel 间的细粒度同步。

.. py:function:: gdc_launch_dependents()

   在当前 kernel 内部触发依赖它的后继 kernel 启动。
   调用此函数后，后继 kernel 即可开始执行，无需等待当前 kernel 完全结束。

详情请参阅 `官方文档 <https://triton-lang.org/main/python-api/triton.language.extra.cuda.html>`_。
