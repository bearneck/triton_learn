================
调试 Triton
================

本教程为调试 Triton 程序提供指导，主要面向 Triton 用户。
对于希望深入了解 Triton 后端（包括 MLIR 代码变换和 LLVM 代码生成）的开发者，
可参考 `此章节 <https://github.com/triton-lang/triton?tab=readme-ov-file#tips-for-hacking>`_ 了解相关调试选项。

------------------------------------
使用 Triton 内置调试算子
------------------------------------

Triton 提供了四个调试算子，供用户检查和查看张量值：

- :code:`static_print` 和 :code:`static_assert` 用于编译期调试。
- :code:`device_print` 和 :code:`device_assert` 用于运行期调试。

:code:`device_assert` 仅在 :code:`TRITON_DEBUG` 被设置为 :code:`1` 时才会执行。
其他调试算子无论 :code:`TRITON_DEBUG` 取何值都会执行。

----------------------------
使用解释器
----------------------------

解释器是调试 Triton 程序的一种简单实用的工具。
它允许用户在 CPU 上运行 Triton 程序，并逐步检查每个操作的中间结果。
将环境变量 :code:`TRITON_INTERPRET` 设置为 :code:`1` 即可启用解释器模式。
该设置会使所有 Triton kernel 跳过编译，转而由解释器使用 Triton 操作对应的 numpy 等价函数进行模拟。
解释器按顺序逐一处理每个 Triton 程序实例，依次执行各操作。

使用解释器主要有以下三种方式：

- 使用 Python :code:`print` 函数打印每个操作的中间结果。查看整个张量使用 :code:`print(tensor)`；查看索引 :code:`idx` 处的张量值使用 :code:`print(tensor.handle.data[idx])`。

- 附加 :code:`pdb` 对 Triton 程序进行逐步调试：

  .. code-block:: bash

    TRITON_INTERPRET=1 pdb main.py
    b main.py:<行号>
    r

- 在 Triton 程序中导入 :code:`pdb` 包并设置断点：

  .. code-block:: python

    import triton
    import triton.language as tl
    import pdb

    @triton.jit
    def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
      pdb.set_trace()
      offs = tl.arange(0, BLOCK_SIZE)
      x = tl.load(x_ptr + offs)
      tl.store(y_ptr + offs, x)

++++++++++++++++++
已知限制
++++++++++++++++++

解释器存在以下几个已知限制：

- 不支持 :code:`bfloat16` 数值类型的操作。如需对 :code:`bfloat16` 张量进行操作，请使用 :code:`tl.cast(tensor)` 将其转换为 :code:`float32`。
- 不支持间接内存访问模式，例如：

  .. code-block:: python

    ptr = tl.load(ptr)
    x = tl.load(ptr)

----------------------------
使用第三方工具
----------------------------

在 NVIDIA GPU 上调试时，`compute-sanitizer <https://docs.nvidia.com/cuda/compute-sanitizer/index.html>`_ 是检查数据竞争和内存访问问题的有效工具。
使用时，在运行 Triton 程序的命令前加上 :code:`compute-sanitizer` 即可。

在 AMD GPU 上调试时，可以尝试使用适用于 ROCm 的 LLVM `AddressSanitizer <https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html>`_。

如需对 Triton 程序的内存访问进行详细可视化，可以考虑使用 `triton-viz <https://github.com/Deep-Learning-Profiling-Tools/triton-viz>`_ 工具，该工具与底层 GPU 架构无关。
