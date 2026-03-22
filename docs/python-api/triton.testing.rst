triton.testing
==============

``triton.testing`` 模块提供了用于性能基准测试的工具。

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 名称
     - 说明
   * - ``Benchmark``
     - 定义一组基准测试的配置类，描述 x 轴变量、对比线条等
   * - ``do_bench``
     - 对一个 Python callable 进行计时，返回中位数/最小/最大运行时间（毫秒）
   * - ``do_bench_cudagraph``
     - 通过 CUDA Graph 捕获后再计时，减少 CPU 开销影响
   * - ``perf_report``
     - 装饰器，将 ``Benchmark`` 配置与基准函数绑定，支持绘图和打印
   * - ``assert_close``
     - 断言两个张量在数值上近似相等

详情请参阅 `官方 triton.testing API 文档 <https://triton-lang.org/main/python-api/triton.testing.html>`_。
