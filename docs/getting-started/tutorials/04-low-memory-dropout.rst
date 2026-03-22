低内存 Dropout（Low-Memory Dropout）
======================================

在本教程中，你将编写一个内存效率高于 PyTorch 原生实现的 dropout 算子。

通过本教程，你将学到：

* 在 Triton 中实现原语（primitives）的子元素随机性（sub-element randomness）。

* Triton 中伪随机数生成器（PRNG）的性能特性。

* 在 Triton 中实现种子可复现的随机化（seeded randomization）的考量。

.. note::

   `查看源码 <https://github.com/bearneck/triton_learn/blob/master/tutorials/04-low-memory-dropout.py>`_

基本原理
----------

Dropout（随机失活）是一种正则化技术，通过以概率 :math:`p` 随机地将输入张量的元素置零，
并将其余元素缩放 :math:`\frac{1}{1-p}`，从而在训练神经网络时防止过拟合。

朴素实现方式会分配一个均匀分布的随机数矩阵，与输入进行比较，
以确定哪些元素应被置零，这需要读写与输入等大的内存空间。
下面将展示 Triton 的实现方式，将掩码生成融入 kernel 本身，
无需额外的内存读写。

.. code-block:: python

   import tabulate
   import torch

   import triton
   import triton.language as tl

   DEVICE = triton.runtime.driver.active.get_active_torch_device()

计算 Kernel（核函数）
-----------------------

dropout kernel 接受如下参数：
 - `x_ptr`：指向输入张量的指针
 - `output_ptr`：指向输出张量的指针
 - `n_elements`：张量中的元素数量
 - `p`：某个元素被置零的概率
 - `seed`：随机数生成器的种子

在此 kernel 中，每个程序实例处理 BLOCK_SIZE 个连续的元素。
对于每个元素，使用 `tl.rand` 生成一个均匀分布的随机数，
若该随机数大于 `p`，则保留该元素（并按比例缩放），否则将其置零。

.. code-block:: python

   @triton.jit
   def _dropout(
       x_ptr,  # 指向输入张量的指针
       x_keep_ptr,  # 指向掩码张量的指针
       output_ptr,  # 指向输出张量的指针
       n_elements,  # `x` 张量中的元素数量
       p,  # dropout 的丢弃概率（每个元素被置零的概率）
       BLOCK_SIZE: tl.constexpr,
   ):
       pid = tl.program_id(axis=0)
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)
       x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
       output = tl.where(x_keep, x / (1 - p), 0.0)
       tl.store(output_ptr + offsets, output, mask=mask)

   def dropout(x, x_keep, p):
       output = torch.empty_like(x)
       assert x.is_contiguous()
       n_elements = x.numel()
       grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
       _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
       return output

输入张量及随机掩码的准备

.. code-block:: python

   torch.manual_seed(0)
   x = torch.randn(size=(10, ), device=DEVICE)
   p = 0.5
   x_keep = (torch.rand(size=(10, ), device=DEVICE) > p).to(torch.int32)
   output = dropout(x, x_keep=x_keep, p=p)
   print(tabulate.tabulate([
       ["input"] + x.tolist(),
       ["keep mask"] + x_keep.tolist(),
       ["output"] + output.tolist(),
   ]))

种子化 Dropout（Seeded Dropout）
----------------------------------

上述 dropout 实现需要存储掩码张量，对于大型模型而言，该额外内存开销不可忽视。
一种更节省内存的实现方式是使用伪随机数生成器（PRNG）在 kernel 内部按需生成随机数，
这样就无需显式存储掩码——只需保存用于重现随机序列的种子（seed）即可。

在本节中，我们实现一个种子化 dropout kernel，利用 Triton 内置的
`tl.rand` 函数即时生成随机掩码，同时支持通过相同的种子和偏移量
在反向传播时精确重现相同的随机数，从而避免存储掩码所带来的内存开销。

.. code-block:: python

   @triton.jit
   def _seeded_dropout(
       x_ptr,
       output_ptr,
       n_elements,
       p,
       seed,
       BLOCK_SIZE: tl.constexpr,
   ):
       pid = tl.program_id(axis=0)
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)
       random = tl.rand(seed, offsets)
       x_keep = random > p
       output = tl.where(x_keep, x / (1 - p), 0.0)
       tl.store(output_ptr + offsets, output, mask=mask)

   def seeded_dropout(x, p, seed):
       output = torch.empty_like(x)
       assert x.is_contiguous()
       n_elements = x.numel()
       grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
       _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
       return output

   x = torch.randn(size=(10, ), device=DEVICE)
   output1 = seeded_dropout(x, p=0.5, seed=123)
   output2 = seeded_dropout(x, p=0.5, seed=123)
   output3 = seeded_dropout(x, p=0.5, seed=512)

   print(
       tabulate.tabulate([
           ["input"] + x.tolist(),
           ["output (seed = 123)"] + output1.tolist(),
           ["output (seed = 123)"] + output2.tolist(),
           ["output (seed = 512)"] + output3.tolist(),
       ]))

符合预期！相同的种子产生相同的 dropout 掩码，不同的种子则产生不同的结果。

延伸练习
----------

本教程所展示的技术，是各种需要高效重随机化（re-randomization）技术的算法的基础，
例如用于训练的随机深度（Stochastic Depth）或随机 Transformer 中的
块状稀疏注意力（blockwise sparse attention）。

若有兴趣，可以尝试扩展上述 kernel，使其支持张量的分块稀疏化，
即：以 block（块）为单位执行 dropout，而非逐元素进行。
