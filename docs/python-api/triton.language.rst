triton.language
===============

.. currentmodule:: triton.language


编程模型
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    tensor_descriptor
    program_id
    num_programs


创建算子
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    arange
    cat
    full
    zeros
    zeros_like
    cast


形状操作算子
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast
    broadcast_to
    expand_dims
    interleave
    join
    permute
    ravel
    reshape
    split
    trans
    view


线性代数算子
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    dot
    dot_scaled


内存/指针算子
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    load
    store
    make_tensor_descriptor
    load_tensor_descriptor
    store_tensor_descriptor
    make_block_ptr
    advance


索引算子
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    flip
    where
    swizzle2d


数学算子
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    cdiv
    ceil
    clamp
    cos
    div_rn
    erf
    exp
    exp2
    fdiv
    floor
    fma
    log
    log2
    maximum
    minimum
    rsqrt
    sigmoid
    sin
    softmax
    sqrt
    sqrt_rn
    umulhi


归约算子
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    argmax
    argmin
    max
    min
    reduce
    sum
    xor_sum

扫描/排序算子
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    associative_scan
    cumprod
    cumsum
    histogram
    sort
    topk
    gather

原子算子
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_add
    atomic_and
    atomic_cas
    atomic_max
    atomic_min
    atomic_or
    atomic_xchg
    atomic_xor

随机数生成
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    randint4x
    randint
    rand
    randn


迭代器
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    range
    static_range


内联汇编
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inline_asm_elementwise


编译器提示算子
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    assume
    debug_barrier
    max_constancy
    max_contiguous
    multiple_of


调试算子
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    static_print
    static_assert
    device_print
    device_assert
