欢迎阅读 Triton 中文文档！
===========================

`Triton <https://github.com/triton-lang/triton>`_ 是一种用于并行编程的语言与编译器。它旨在提供一个基于 Python 的编程环境，帮助开发者高效地编写自定义 DNN 计算 kernel，并在现代 GPU 硬件上以最大吞吐量运行。

.. note::

   本文档为 `Triton 官方文档 <https://triton-lang.org/main/index.html>`_ 及教程的中文翻译版本，代码内容与官方保持一致。

快速入门
--------

- 按照适合你所用平台的 :doc:`安装说明 <getting-started/installation>` 进行安装。
- 查阅 :doc:`教程索引 <getting-started/tutorials/index>`，学习如何编写你的第一个 Triton 程序。

.. toctree::
   :maxdepth: 2
   :caption: 快速入门

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
   :maxdepth: 2
   :caption: Python API

   python-api/triton
   python-api/triton.language
   python-api/triton.testing
   python-api/triton-semantics
   python-api/triton.language.extra.cuda

深入学习
--------

了解 Triton 的设计理念及其与其他 DNN DSL 的对比：

- 第一章：:doc:`简介 <programming-guide/chapter-1/introduction>`
- 第二章：:doc:`相关工作 <programming-guide/chapter-2/related-work>`
- 第三章：:doc:`调试 <programming-guide/chapter-3/debugging>`

.. toctree::
   :maxdepth: 2
   :caption: 编程指南

   programming-guide/chapter-1/introduction
   programming-guide/chapter-2/related-work
   programming-guide/chapter-3/debugging
