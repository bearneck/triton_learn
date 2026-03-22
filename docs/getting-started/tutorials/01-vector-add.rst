向量加法
==========

在本教程中，你将使用 Triton 编写一个简单的向量加法程序。

通过本教程，你将学到：

* Triton 的基本编程模型。

* `triton.jit` 装饰器的用法——用于定义 Triton kernel（核函数）。

* 针对原生参考实现验证和基准测试自定义算子的最佳实践。

.. note::

   `查看源码 <https://github.com/bearneck/triton_learn/blob/master/tutorials/01-vector-add.py>`_

计算 Kernel（核函数）
-----------------------

.. code-block:: python

   import torch

   import triton
   import triton.language as tl

   DEVICE = triton.runtime.driver.active.get_active_torch_device()

   @triton.jit
   def add_kernel(x_ptr,  # 指向第一个输入向量的 *指针*。
                  y_ptr,  # 指向第二个输入向量的 *指针*。
                  output_ptr,  # 指向输出向量的 *指针*。
                  n_elements,  # 向量的大小。
                  BLOCK_SIZE: tl.constexpr,  # 每个 program（程序实例）应处理的元素数量。
                  ):
       pid = tl.program_id(axis=0)  # 我们使用一维启动网格，因此 axis 为 0。
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)
       output = x + y
       tl.store(output_ptr + offsets, output, mask=mask)

下面声明一个辅助函数，用于：(1) 分配 `z` 张量；
(2) 以合适的 grid/block 尺寸将上述 kernel 加入执行队列：

.. code-block:: python

   def add(x: torch.Tensor, y: torch.Tensor):
       output = torch.empty_like(x)
       assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
       n_elements = output.numel()
       grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
       add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
       return output

现在可以使用上述函数计算两个 `torch.tensor` 对象的逐元素之和，并验证其正确性：

.. code-block:: python

   torch.manual_seed(0)
   size = 98432
   x = torch.rand(size, device=DEVICE)
   y = torch.rand(size, device=DEVICE)
   output_torch = x + y
   output_triton = add(x, y)
   print(output_torch)
   print(output_triton)
   print(f'torch 与 triton 之间的最大差值为 '
         f'{torch.max(torch.abs(output_torch - output_triton))}')

结果看起来没问题！

基准测试（Benchmark）
-----------------------

现在对不同大小的向量进行自定义算子的基准测试，以了解其相对于 PyTorch 的性能表现。
为便于操作，Triton 内置了一组实用工具，可简洁地绘制自定义算子在不同问题规模下的性能曲线。

.. code-block:: python

   @triton.testing.perf_report(
       triton.testing.Benchmark(
           x_names=['size'],  # 用作图表 x 轴的参数名。
           x_vals=[2**i for i in range(12, 28, 1)],  # `x_name` 的不同取值。
           x_log=True,  # x 轴使用对数坐标。
           line_arg='provider',  # 其值对应图中不同折线的参数名。
           line_vals=['triton', 'torch'],  # `line_arg` 的可能取值。
           line_names=['Triton', 'Torch'],  # 各折线的标签名称。
           styles=[('blue', '-'), ('green', '-')],  # 折线样式。
           ylabel='GB/s',  # y 轴的标签名称。
           plot_name='vector-add-performance',  # 图表名称，同时也用作保存文件时的文件名。
           args={},  # 不在 `x_names` 和 `y_name` 中的函数参数取值。
       ))
   def benchmark(size, provider):
       x = torch.rand(size, device=DEVICE, dtype=torch.float32)
       y = torch.rand(size, device=DEVICE, dtype=torch.float32)
       quantiles = [0.5, 0.2, 0.8]
       if provider == 'torch':
           ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
       if provider == 'triton':
           ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
       gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
       return gbps(ms), gbps(max_ms), gbps(min_ms)

现在可以运行上面经过装饰的函数。传入 `print_data=True` 可查看性能数据，
`show_plots=True` 可绘制图表，`save_path='/path/to/results/'` 可将结果
连同原始 CSV 数据一起保存到磁盘：

.. code-block:: python

   benchmark.run(print_data=True, show_plots=True)
