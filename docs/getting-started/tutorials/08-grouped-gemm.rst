分组 GEMM
===========

该分组 GEMM kernel 启动固定数量的 CTA 来计算一组 GEMM。
调度策略为静态，并在设备端执行。

.. note::

   `查看源码 <https://github.com/bearneck/triton_learn/blob/master/tutorials/08-grouped-gemm.py>`_
