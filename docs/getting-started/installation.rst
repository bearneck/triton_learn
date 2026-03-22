============
安装
============

有关支持的平台/操作系统及硬件，请查阅 Github 上的 `兼容性 <https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility>`_ 章节。

--------------------
二进制发行版
--------------------

你可以通过 pip 安装最新稳定版 Triton：

.. code-block:: bash

      pip install triton

提供适用于 CPython 3.10-3.14 的二进制 wheel 包。

-----------
从源码安装
-----------

++++++++++++++
Python 包
++++++++++++++

你可以通过以下命令从源码安装 Python 包：

.. code-block:: bash

      git clone https://github.com/triton-lang/triton.git
      cd triton

      pip install -r python/requirements.txt # 构建时依赖
      pip install -e .

注意：如果你的系统中没有安装 LLVM，setup.py 脚本将自动下载官方 LLVM 静态库并进行链接。

如需使用自定义 LLVM 进行构建，请参阅 Github 上的 `使用自定义 LLVM 构建 <https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm>`_ 章节。

安装完成后，你可以运行测试来验证安装是否正常：

.. code-block:: bash

      # 一次性初始化
      make dev-install

      # 运行全部测试（需要 GPU）
      make test

      # 或者，运行不需要 GPU 的测试
      make test-nogpu
