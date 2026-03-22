triton
======

.. currentmodule:: triton

.. py:function:: jit(fn)

   使用 Triton 编译器对函数进行 JIT 编译的装饰器。被装饰的函数将在首次调用时编译为 GPU kernel，并可通过 ``fn[grid](args...)`` 的方式启动。

.. py:function:: autotune(configs, key, ...)

   对 ``triton.jit`` 修饰的函数进行自动调优的装饰器。给定一组候选配置（``configs``），在 ``key`` 参数变化时自动选择最优配置。

.. py:function:: heuristics(values)

   指定某些 meta 参数取值方式的装饰器。允许根据输入参数的启发式规则动态计算 ``constexpr`` 参数，而无需穷举所有配置。

.. py:class:: Config(kwargs, num_warps=4, num_stages=2, ...)

   表示自动调优器要尝试的一种 kernel 配置。封装了 meta 参数字典、warp 数量、流水线阶段数等信息。

详情请参阅 `官方 triton API 文档 <https://triton-lang.org/main/python-api/triton.html>`_。
