欢迎阅读 Triton 文档！
==================================

Triton_ 是一种用于并行编程的语言与编译器。它旨在提供一个基于 Python 的编程环境，帮助开发者高效地编写自定义 DNN 计算 kernel，并在现代 GPU 硬件上以最大吞吐量运行。


快速入门
---------------

- 按照适合你所用平台的 :doc:`安装说明 <getting-started/installation>` 进行安装。
- 查阅 :doc:`教程 <getting-started/tutorials/index>`，学习如何编写你的第一个 Triton 程序。

.. toctree::
   :maxdepth: 1
   :caption: 快速入门
   :hidden:

   getting-started/installation
   getting-started/tutorials/index


Python API
----------

- :doc:`triton <python-api/triton>`
- :doc:`triton.language <python-api/triton.language>`
- :doc:`triton.testing <python-api/triton.testing>`
- :doc:`Triton 语义 <python-api/triton-semantics>`
- :doc:`triton.language.extra.cuda <python-api/triton.language.extra.cuda>`


.. toctree::
   :maxdepth: 1
   :caption: Python API
   :hidden:

   python-api/triton
   python-api/triton.language
   python-api/triton.testing
   python-api/triton-semantics


Triton MLIR 方言与算子
--------------------

- :doc:`Triton MLIR 方言与算子 <dialects/dialects>`

.. toctree::
   :maxdepth: 1
   :caption: Triton MLIR 方言
   :hidden:

   dialects/dialects

深入学习
-------------

阅读以下文档，深入了解 Triton 及其与其他 DNN DSL 的对比：

- 第一章：:doc:`简介 <programming-guide/chapter-1/introduction>`
- 第二章：:doc:`相关工作 <programming-guide/chapter-2/related-work>`
- 第三章：:doc:`调试 <programming-guide/chapter-3/debugging>`

.. toctree::
   :maxdepth: 1
   :caption: 编程指南
   :hidden:

   programming-guide/chapter-1/introduction
   programming-guide/chapter-2/related-work
   programming-guide/chapter-3/debugging

.. _Triton: https://github.com/triton-lang/triton
