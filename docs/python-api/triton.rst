triton
======

以下是 ``triton`` 模块的核心接口。

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 名称
     - 说明
   * - ``triton.jit``
     - 将 Python 函数编译为 Triton GPU kernel 的装饰器
   * - ``triton.autotune``
     - 对 kernel 进行自动性能调优的装饰器，遍历给定配置列表
   * - ``triton.heuristics``
     - 基于启发式规则动态选择 kernel 参数的装饰器
   * - ``triton.Config``
     - 用于 ``autotune`` 的配置对象，封装 meta 参数、warp 数量、流水线阶段数等

详情请参阅 `官方 triton API 文档 <https://triton-lang.org/main/python-api/triton.html>`_。
