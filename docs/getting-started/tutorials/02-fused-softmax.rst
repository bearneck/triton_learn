融合 Softmax（Fused Softmax）
===============================

在本教程中，你将编写一个融合的 softmax 操作，对于特定类别的矩阵——
其行能够放入 GPU 的 SRAM（片上内存）——其速度将显著快于 PyTorch 的原生算子。

通过本教程，你将学到：

* kernel 融合（kernel fusion）对带宽受限操作的优势。

* Triton 中的归约（reduction）算子。

.. note::

   `查看源码 <https://github.com/bearneck/triton_learn/blob/master/tutorials/02-fused-softmax.py>`_

动机
----------

针对逐元素加法编写自定义 GPU kernel 具有教学价值，但在实际中用处有限。
让我们转而考虑一个简单的（数值稳定的）softmax 操作：

.. code-block:: python

   import torch

   import triton
   import triton.language as tl
   from triton.runtime import driver

   DEVICE = triton.runtime.driver.active.get_active_torch_device()

   def is_hip():
       return triton.runtime.driver.active.get_current_target().backend == "hip"

   def is_cdna():
       return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                      'gfx90a', 'gfx908')

   def naive_softmax(x):
       """使用原生 PyTorch 计算 X 的按行 softmax。

       为避免数值溢出，我们减去每行的最大值。softmax 对该平移操作保持不变。
       """
       x_max = x.max(dim=1)[0]
       z = x - x_max[:, None]
       numerator = torch.exp(z)
       denominator = numerator.sum(dim=1)
       ret = numerator / denominator[:, None]
       return ret

在 PyTorch 中朴素实现时，对 :math:`x \in R^{M \times N}` 计算
:code:`y = naive_softmax(x)` 需要从 DRAM 读取 :math:`5MN + 2M` 个元素，
并写回 :math:`3MN + 2M` 个元素。这显然是一种浪费；我们更希望有一个自定义的
"融合" kernel，只读取一次 X，并在片上完成所有必要的计算。
这样只需读写 :math:`MN` 字节，理论上可获得约 4 倍的加速
（即 :math:`(8MN + 4M) / 2MN`）。
`torch.jit.script` 标志旨在自动执行这种 "kernel 融合"，
但正如我们稍后将看到的，它离理想状态仍有差距。

计算 Kernel（核函数）
-----------------------

我们的 softmax kernel 工作方式如下：每个 program 按 program 总数为步长，
加载输入矩阵 X 的若干行，对其进行归一化，然后将结果写回输出矩阵 Y。

需要注意 Triton 的一个重要限制：每个 block 的元素数量必须是 2 的幂次，
因此若要处理任意形状的输入，需要在内部对每行进行 "填充"（pad），
并妥善处理内存操作的边界保护：

.. code-block:: python

   @triton.jit
   def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                      num_stages: tl.constexpr):
       row_start = tl.program_id(0)
       row_step = tl.num_programs(0)
       for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
           row_start_ptr = input_ptr + row_idx * input_row_stride
           col_offsets = tl.arange(0, BLOCK_SIZE)
           input_ptrs = row_start_ptr + col_offsets
           mask = col_offsets < n_cols
           row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
           row_minus_max = row - tl.max(row, axis=0)
           numerator = tl.exp(row_minus_max)
           denominator = tl.sum(numerator, axis=0)
           softmax_output = numerator / denominator
           output_row_start_ptr = output_ptr + row_idx * output_row_stride
           output_ptrs = output_row_start_ptr + col_offsets
           tl.store(output_ptrs, softmax_output, mask=mask)

我们可以创建一个辅助函数，对任意输入张量将 kernel 及其（元）参数加入执行队列。

.. code-block:: python

   properties = driver.active.utils.get_device_properties(DEVICE.index)
   NUM_SM = properties["multiprocessor_count"]
   NUM_REGS = properties["max_num_regs"]
   SIZE_SMEM = properties["max_shared_mem"]
   WARP_SIZE = properties["warpSize"]
   target = triton.runtime.driver.active.get_current_target()
   kernels = {}

   def softmax(x):
       n_rows, n_cols = x.shape

       BLOCK_SIZE = triton.next_power_of_2(n_cols)

       num_warps = 8

       num_stages = 4 if SIZE_SMEM > 200000 else 2

       y = torch.empty_like(x)

       kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                      num_stages=num_stages, num_warps=num_warps, grid=(1, ))
       kernel._init_handles()
       n_regs = kernel.n_regs
       size_smem = kernel.metadata.shared
       if is_hip():
           NUM_GPRS = NUM_REGS
           if is_cdna():
               NUM_GPRS = NUM_REGS * 2
           MAX_NUM_THREADS = properties["max_threads_per_sm"]
           max_num_waves = MAX_NUM_THREADS // WARP_SIZE
           occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
       else:
           occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
       occupancy = min(occupancy, SIZE_SMEM // size_smem)
       num_programs = NUM_SM * occupancy

       num_programs = min(num_programs, n_rows)

       kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
       return y

单元测试
----------

使用行列数不规则的矩阵测试我们的 kernel，以验证填充（padding）机制是否正确。

.. code-block:: python

   torch.manual_seed(0)
   x = torch.randn(1823, 781, device=DEVICE)
   y_triton = softmax(x)
   y_torch = torch.softmax(x, axis=1)
   assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

结果与预期一致。

基准测试（Benchmark）
-----------------------

下面以输入矩阵的列数为变量（假设行数为 4096）对该操作进行基准测试，
并将其与 (1) :code:`torch.softmax` 和 (2) 上面定义的 :code:`naive_softmax` 进行性能对比。

.. code-block:: python

   @triton.testing.perf_report(
       triton.testing.Benchmark(
           x_names=['N'],  # 用作图表 x 轴的参数名
           x_vals=[128 * i for i in range(2, 100)],  # `x_name` 的不同取值
           line_arg='provider',  # 其值对应图中不同折线的参数名
           line_vals=['triton', 'torch', 'naive_softmax'],  # `line_arg` 的可能取值
           line_names=["Triton", "Torch", "Naive Softmax"],  # 各折线的标签名称
           styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # 折线样式
           ylabel="GB/s",  # y 轴的标签名称
           plot_name="softmax-performance",  # 图表名称，同时也用作保存文件时的文件名
           args={'M': 4096},  # 不在 `x_names` 和 `y_name` 中的函数参数取值
       ))
   def benchmark(M, N, provider):
       x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
       stream = getattr(torch, DEVICE.type).Stream()
       getattr(torch, DEVICE.type).set_stream(stream)
       if provider == 'torch':
           ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
       if provider == 'triton':
           ms = triton.testing.do_bench(lambda: softmax(x))
       if provider == 'naive_softmax':
           ms = triton.testing.do_bench(lambda: naive_softmax(x))
       gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
       return gbps(ms)

   benchmark.run(show_plots=True, print_data=True)

从上图可以观察到：
 - Triton 比 Torch JIT 快 4 倍，这证实了我们的判断：Torch JIT 在此处未进行任何融合。
 - Triton 明显快于 :code:`torch.softmax`——同时**更易于阅读、理解和维护**。
   但需注意，PyTorch 的 `softmax` 操作更为通用，可处理任意形状的张量。
