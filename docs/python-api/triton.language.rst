triton.language
===============

``triton.language``（简称 ``tl``）是编写 Triton kernel 时使用的核心语言模块。

编程模型
--------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - 名称
     - 说明
   * - ``tl.tensor``
     - Triton 的基本数据类型，表示一个多维张量块
   * - ``tl.tensor_descriptor``
     - TMA 张量描述符类型
   * - ``tl.program_id(axis)``
     - 返回当前程序实例在指定轴上的 ID
   * - ``tl.num_programs(axis)``
     - 返回指定轴上程序实例的总数

创建算子
--------

``arange`` · ``cat`` · ``full`` · ``zeros`` · ``zeros_like`` · ``cast``

形状操作算子
------------

``broadcast`` · ``broadcast_to`` · ``expand_dims`` · ``interleave`` · ``join`` · ``permute`` · ``ravel`` · ``reshape`` · ``split`` · ``trans`` · ``view``

线性代数算子
------------

``dot`` · ``dot_scaled``

内存 / 指针算子
---------------

``load`` · ``store`` · ``make_tensor_descriptor`` · ``load_tensor_descriptor`` · ``store_tensor_descriptor`` · ``make_block_ptr`` · ``advance``

索引算子
--------

``flip`` · ``where`` · ``swizzle2d``

数学算子
--------

``abs`` · ``cdiv`` · ``ceil`` · ``clamp`` · ``cos`` · ``div_rn`` · ``erf`` · ``exp`` · ``exp2`` · ``fdiv`` · ``floor`` · ``fma`` · ``log`` · ``log2`` · ``maximum`` · ``minimum`` · ``rsqrt`` · ``sigmoid`` · ``sin`` · ``softmax`` · ``sqrt`` · ``sqrt_rn`` · ``umulhi``

归约算子
--------

``argmax`` · ``argmin`` · ``max`` · ``min`` · ``reduce`` · ``sum`` · ``xor_sum``

扫描 / 排序算子
---------------

``associative_scan`` · ``cumprod`` · ``cumsum`` · ``histogram`` · ``sort`` · ``topk`` · ``gather``

原子算子
--------

``atomic_add`` · ``atomic_and`` · ``atomic_cas`` · ``atomic_max`` · ``atomic_min`` · ``atomic_or`` · ``atomic_xchg`` · ``atomic_xor``

随机数生成
----------

``randint4x`` · ``randint`` · ``rand`` · ``randn``

迭代器
------

``range`` · ``static_range``

内联汇编
--------

``inline_asm_elementwise``

编译器提示算子
--------------

``assume`` · ``debug_barrier`` · ``max_constancy`` · ``max_contiguous`` · ``multiple_of``

调试算子
--------

``static_print`` · ``static_assert`` · ``device_print`` · ``device_assert``

详情请参阅 `官方 triton.language API 文档 <https://triton-lang.org/main/python-api/triton.language.html>`_。
