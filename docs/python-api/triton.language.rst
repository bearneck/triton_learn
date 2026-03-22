triton.language
===============

.. currentmodule:: triton.language

``triton.language``（简称 ``tl``）是编写 Triton kernel 时使用的核心语言模块。

编程模型
--------

.. py:class:: tensor

   表示一个 N 维的值或指针数组，是 Triton 的基本数据类型。

.. py:class:: tensor_descriptor

   表示全局内存中张量的描述符对象，用于 TMA（Tensor Memory Accelerator）访问。

.. py:function:: program_id(axis)

   返回当前程序实例在指定轴（axis）上的 ID。

.. py:function:: num_programs(axis)

   返回沿指定轴启动的程序实例总数。

创建算子
--------

.. py:function:: arange(start, end)

   返回半开区间 ``[start, end)`` 内的连续整数值张量。

.. py:function:: cat(x, y, can_reorder=False)

   沿最后一维拼接两个张量块。

.. py:function:: full(shape, value, dtype)

   返回指定形状和 dtype、填充标量值 ``value`` 的张量。

.. py:function:: zeros(shape, dtype)

   返回指定形状和 dtype、填充标量值 ``0`` 的张量。

.. py:function:: zeros_like(input)

   返回与给定张量形状和类型相同的全零张量。

.. py:function:: cast(input, dtype, fp_downcast_rounding=None, bitcast=False)

   将张量转换为指定 dtype。

形状操作算子
------------

.. py:function:: broadcast(input, other)

   尝试将两个张量块广播到兼容的公共形状。

.. py:function:: broadcast_to(input, shape)

   尝试将给定张量广播到新形状。

.. py:function:: expand_dims(input, axis)

   通过在指定位置插入长度为 1 的新维度来扩展张量形状。

.. py:function:: interleave(a, b)

   沿最后一维交错排列两个张量的值。

.. py:function:: join(a, b)

   在一个新的末尾维度上拼接两个张量。

.. py:function:: permute(x, dims)

   对张量的维度进行排列变换。

.. py:function:: ravel(x)

   返回 x 的连续展平视图。

.. py:function:: reshape(x, shape)

   返回与输入元素数量相同但形状不同的张量。

.. py:function:: split(x)

   沿最后一维（大小必须为 2）将张量拆分为两个张量。

.. py:function:: trans(x, *dims)

   对张量的维度进行转置排列。

.. py:function:: view(x, shape)

   返回与输入元素相同但形状不同的张量视图。

线性代数算子
------------

.. py:function:: dot(input, other, acc=None, input_precision=None, max_num_imprecise_acc=None, out_dtype=tl.float32)

   返回两个张量块的矩阵乘积。

.. py:function:: dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, out_dtype=tl.float32, fast_math=True)

   返回两个微缩放格式（microscaling format）张量块的矩阵乘积。

内存 / 指针算子
---------------

.. py:function:: load(pointer, mask=None, other=None, boundary_check=(), padding_option='', cache_modifier='', eviction_policy='', volatile=False)

   从 ``pointer`` 指定的内存位置加载数据，返回张量。支持掩码和越界检查。

.. py:function:: store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='')

   将张量数据存储到 ``pointer`` 指定的内存位置。

.. py:function:: make_tensor_descriptor(base, shape, strides, block_shape)

   创建张量描述符对象，用于 TMA 访问。

.. py:function:: load_tensor_descriptor(desc, indices)

   从张量描述符加载一块数据。

.. py:function:: store_tensor_descriptor(desc, indices, value)

   将一块数据存储到张量描述符指定的位置。

.. py:function:: make_block_ptr(base, shape, strides, offsets, block_shape, order)

   返回指向父张量中某个块的指针。

.. py:function:: advance(base, offsets)

   推进块指针的偏移量。

索引算子
--------

.. py:function:: flip(x, dim=None)

   沿指定维度 ``dim`` 翻转张量 x。

.. py:function:: where(condition, x, y)

   根据条件张量，逐元素从 x 或 y 中选取值，返回结果张量。

.. py:function:: swizzle2d(i, j, size_i, size_j, size_g)

   将行优先 ``size_i × size_j`` 矩阵的索引变换为每组 ``size_g`` 行的列优先矩阵索引，提升 L2 cache 局部性。

数学算子
--------

.. py:function:: abs(x)

   逐元素计算 x 的绝对值。

.. py:function:: cdiv(x, div)

   计算 x 除以 div 的向上取整（ceiling division）。

.. py:function:: ceil(x)

   逐元素计算 x 的向上取整。

.. py:function:: clamp(x, min, max)

   将输入张量 x 的值截断到区间 ``[min, max]`` 内。

.. py:function:: cos(x)

   逐元素计算 x 的余弦值。

.. py:function:: div_rn(x, y)

   逐元素计算 x / y，并按 IEEE 标准向最近值舍入（精确除法）。

.. py:function:: erf(x)

   逐元素计算 x 的误差函数。

.. py:function:: exp(x)

   逐元素计算 e 的 x 次幂。

.. py:function:: exp2(x)

   逐元素计算 2 的 x 次幂。

.. py:function:: fdiv(x, y)

   逐元素计算 x / y 的快速除法（可能牺牲精度）。

.. py:function:: floor(x)

   逐元素计算 x 的向下取整。

.. py:function:: fma(x, y, z)

   逐元素计算融合乘加：``x * y + z``。

.. py:function:: log(x)

   逐元素计算 x 的自然对数。

.. py:function:: log2(x)

   逐元素计算 x 的以 2 为底的对数。

.. py:function:: maximum(x, y)

   逐元素计算 x 和 y 的最大值。

.. py:function:: minimum(x, y)

   逐元素计算 x 和 y 的最小值。

.. py:function:: rsqrt(x)

   逐元素计算 x 的平方根的倒数（1 / sqrt(x)）。

.. py:function:: sigmoid(x)

   逐元素计算 x 的 sigmoid 函数值。

.. py:function:: sin(x)

   逐元素计算 x 的正弦值。

.. py:function:: softmax(x, ieee_rounding=False)

   逐元素计算 x 的 softmax 值。

.. py:function:: sqrt(x)

   逐元素计算 x 的快速平方根。

.. py:function:: sqrt_rn(x)

   逐元素计算 x 的精确平方根，按 IEEE 标准向最近值舍入。

.. py:function:: umulhi(x, y)

   逐元素计算 x 和 y 乘积（2N 位）的高 N 位。

归约算子
--------

.. py:function:: argmax(input, axis, keep_dims=False)

   返回输入张量沿指定轴方向最大值的索引。

.. py:function:: argmin(input, axis, keep_dims=False)

   返回输入张量沿指定轴方向最小值的索引。

.. py:function:: max(input, axis, keep_dims=False, return_indices=False, return_indices_tie_break_left=True)

   返回输入张量沿指定轴方向的最大值。

.. py:function:: min(input, axis, keep_dims=False)

   返回输入张量沿指定轴方向的最小值。

.. py:function:: reduce(input, axis, combine_fn, keep_dims=False)

   对输入张量沿指定轴应用自定义 ``combine_fn`` 进行归约。

.. py:function:: sum(input, axis, keep_dims=False)

   返回输入张量沿指定轴方向所有元素的求和结果。

.. py:function:: xor_sum(input, axis, keep_dims=False)

   返回输入张量沿指定轴方向所有元素的按位异或求和结果。

扫描 / 排序算子
---------------

.. py:function:: associative_scan(input, axis, combine_fn, reverse=False)

   沿指定轴对输入张量的每个元素应用带进位的 ``combine_fn``，并更新进位（前缀扫描）。

.. py:function:: cumprod(input, axis, reverse=False)

   返回输入张量沿指定轴方向的累积乘积。

.. py:function:: cumsum(input, axis, reverse=False)

   返回输入张量沿指定轴方向的累积求和结果。

.. py:function:: histogram(input, num_bins)

   基于输入张量计算直方图，共 ``num_bins`` 个桶，桶宽为 1，从 0 开始。

.. py:function:: sort(x, dim=None, descending=False)

   沿指定维度对张量排序，返回排序后的张量。

.. py:function:: topk(input, k, dim=None, largest=True)

   返回输入张量沿指定维度最大（或最小）的 k 个元素。

.. py:function:: gather(src, indices, axis)

   沿指定维度从张量中按索引收集元素。

原子算子
--------

.. py:function:: atomic_add(pointer, val, mask=None, sem=None, scope=None)

   在 ``pointer`` 指定的内存位置执行原子加法操作。

.. py:function:: atomic_and(pointer, val, mask=None, sem=None, scope=None)

   在 ``pointer`` 指定的内存位置执行原子按位与操作。

.. py:function:: atomic_cas(pointer, cmp, val, sem=None, scope=None)

   在 ``pointer`` 指定的内存位置执行原子比较并交换（CAS）操作。

.. py:function:: atomic_max(pointer, val, mask=None, sem=None, scope=None)

   在 ``pointer`` 指定的内存位置执行原子求最大值操作。

.. py:function:: atomic_min(pointer, val, mask=None, sem=None, scope=None)

   在 ``pointer`` 指定的内存位置执行原子求最小值操作。

.. py:function:: atomic_or(pointer, val, mask=None, sem=None, scope=None)

   在 ``pointer`` 指定的内存位置执行原子按位或操作。

.. py:function:: atomic_xchg(pointer, val, mask=None, sem=None, scope=None)

   在 ``pointer`` 指定的内存位置执行原子交换操作。

.. py:function:: atomic_xor(pointer, val, mask=None, sem=None, scope=None)

   在 ``pointer`` 指定的内存位置执行原子按位异或操作。

随机数生成
----------

.. py:function:: randint4x(seed, offset, n_rounds=8)

   给定标量种子和偏移量块，返回四个 int32 随机数块。

.. py:function:: randint(seed, offset, n_rounds=8)

   给定标量种子和偏移量块，返回一个 int32 随机数块。

.. py:function:: rand(seed, offset, n_rounds=8)

   给定标量种子和偏移量块，返回均匀分布在 [0, 1) 区间内的 float32 随机数块。

.. py:function:: randn(seed, offset, n_rounds=8)

   给定标量种子和偏移量块，返回标准正态分布的 float32 随机数块。

迭代器
------

.. py:function:: range(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, flatten=None, disallow_acc_multi_buffer=None)

   向上计数的迭代器，支持循环流水线参数配置。

.. py:function:: static_range(arg1, arg2=None, step=None)

   编译期静态展开的向上计数迭代器，范围须为编译期常量。

内联汇编
--------

.. py:function:: inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack, _builder=None)

   对张量逐元素执行内联汇编指令。

编译器提示算子
--------------

.. py:function:: assume(cond)

   向编译器声明条件 ``cond`` 恒为 True，允许编译器据此进行优化。

.. py:function:: debug_barrier()

   插入屏障以同步同一块内的所有线程（仅用于调试）。

.. py:function:: max_constancy(input, values)

   告知编译器 ``input`` 中前 ``values`` 个值是常量，帮助编译器优化内存访问。

.. py:function:: max_contiguous(input, values)

   告知编译器 ``input`` 中前 ``values`` 个值是连续的，帮助编译器生成向量化访问指令。

.. py:function:: multiple_of(input, values)

   告知编译器 ``input`` 中的值均为 ``values`` 的倍数，帮助编译器优化对齐访问。

调试算子
--------

.. py:function:: static_print(*values, sep=' ', end='\n', file=None, flush=False)

   在编译期打印值（不产生运行时开销）。

.. py:function:: static_assert(cond, msg='')

   在编译期断言条件 ``cond`` 为真，不满足则报错。

.. py:function:: device_print(prefix, *args, hex=False)

   在运行时从 GPU 设备端打印值，用于调试 kernel 中的中间结果。

.. py:function:: device_assert(cond, msg='')

   在运行时断言条件 ``cond`` 为真，需设置 ``TRITON_DEBUG=1`` 才会执行。

详情请参阅 `官方 triton.language API 文档 <https://triton-lang.org/main/python-api/triton.language.html>`_。
