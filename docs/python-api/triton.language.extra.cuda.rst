triton.language.extra.cuda
==========================

``triton.language.extra.cuda`` 提供了 CUDA 特有的扩展功能。

程序化依赖启动
--------------

程序化依赖启动（Programmatic Dependent Launch，PDL）允许 GPU kernel 在完成部分工作后
立即触发下一个 kernel，而无需返回 CPU，从而减少启动延迟。

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 名称
     - 说明
   * - ``gdc_wait``
     - 在 kernel 内等待前驱 kernel 满足依赖条件后再继续执行
   * - ``gdc_launch_dependents``
     - 在当前 kernel 内触发依赖它的后继 kernel 启动

详情请参阅 `官方文档 <https://triton-lang.org/main/python-api/triton.language.extra.cuda.html>`_。
