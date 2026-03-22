triton.testing
==============

.. currentmodule:: triton.testing

``triton.testing`` 模块提供了用于性能基准测试的工具。

.. py:class:: Benchmark(x_names, x_vals, line_arg, line_vals, line_names, ylabel, plot_name, args, x_log=False, y_log=False, styles=None)

   定义一组基准测试的配置类。

   - ``x_names``：用作 x 轴的参数名列表。
   - ``x_vals``：x 轴参数的取值列表。
   - ``line_arg``：其值对应图中不同折线的参数名。
   - ``line_vals``：``line_arg`` 的可能取值列表。
   - ``line_names``：各折线的显示标签。
   - ``ylabel``：y 轴标签。
   - ``plot_name``：图表名称，同时用作保存文件名。
   - ``args``：其他固定参数的字典。

.. py:function:: do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode='mean')

   对一个 Python callable 进行计时，返回运行时间（毫秒）。
   默认返回中位数；通过 ``quantiles`` 参数可同时返回最小值、最大值等分位数。

.. py:function:: do_bench_cudagraph(fn, rep=100, grad_to_none=None, quantiles=None, return_mode='mean')

   通过 CUDA Graph 捕获后再计时，消除 CPU 启动开销对测量结果的影响。

.. py:function:: perf_report(benchmarks)

   将 ``Benchmark`` 配置与基准函数绑定的装饰器。调用 ``.run(show_plots, print_data, save_path)`` 即可执行并输出结果。

.. py:function:: assert_close(x, y, atol=None, rtol=None)

   断言两个张量在数值上近似相等，支持自定义绝对误差和相对误差容限。

详情请参阅 `官方 triton.testing API 文档 <https://triton-lang.org/main/python-api/triton.testing.html>`_。
