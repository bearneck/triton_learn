"""
块缩放矩阵乘法（Block Scaled Matrix Multiplication）
=====================================================
本教程演示了块缩放矩阵乘法的 Triton 实现，
支持 NVIDIA 和 AMD GPU 上的 FP4 和 FP8 格式。
教程支持 OCP 微缩放格式（mxfp4 和 mxfp8）以及 NVIDIA 的 nvfp4
（NVIDIA GPU）和 mxfp4（AMD GPU）。这些矩阵乘法在计算能力为 10 的
NVIDIA GPU 上使用第五代 Tensor Core 进行硬件加速，在 AMD GPU 上
使用 CDNA4 矩阵核心进行加速。
用户可通过传递 `--format` 参数选择支持的格式，并通过指定矩阵维度
和迭代步长对每种格式进行性能基准测试。

.. code-block:: bash

    # FP4
    python 10-block-scaled-matmul.py --format nvfp4
    python 10-block-scaled-matmul.py --format mxfp4 --K_range 512 8192 --bench

    # FP8
    python 10-block-scaled-matmul.py --format mxfp8 --K_range 8192 16384 --K_step 2048 --bench

未来计划更新本教程以支持混合精度块缩放矩阵乘法。
"""

# %%
# 背景
# ----
# NVIDIA GPU 上的 scale 预洗牌（scale preshuffling）
#
# 支持 PTX 8.7 及更高版本的 CUDA 设备可以使用块缩放矩阵乘法指令。
# 为了在 tensor core MMA 快速内循环中以低延迟访问这些缩放因子，
# 需要确保块缩放因子按照其访问模式存储在连续的内存布局中。
#
# 块缩放矩阵乘法 tensor core 指令计算如下乘积：
#
#     C = (A * scale_a) @ (B * scale_b)
#
# 其中 scale_a 和 scale_b 是 A 和 B 矩阵的块缩放因子。
# 在块缩放矩阵乘法中，每个缩放因子会在 A 和 B 矩阵的一组元素上广播并相乘，
# 通常沿各自的 K 轴进行。A 和 B 中每个缩放因子广播覆盖的元素数量
# 在本文中称为向量大小（VEC_SIZE）。
#
# 在线性行主序布局中，缩放因子在全局内存中的形状为：
#
#     (M, K // VEC_SIZE) 和 (N, K // VEC_SIZE)   [1]
#
# 然而，为了避免非连续内存访问，更好的做法是将缩放因子存储在打包块布局中。
# 对于 LHS 矩阵，该布局为：
#
#     (M // 32 // 4, K // VEC_SIZE // 4, 32, 4, 4)   [2]
#
# 这样，在 K 块快速内循环中，每次 tensor core MMA 都可以对矩阵 A 的
# BLOCK_M x BLOCK_K 子块，实现 M 轴上 128 行缩放因子的连续访问。
#
# 为了符合 Triton 语言 dot_scaled 的语义，缩放因子以上述 5D 布局 [2] 准备，
# 然后逻辑上转置并重塑为 tl.dot_scaled 期望的 2D 布局 [1]。
#
# 关于缩放因子布局的更详细信息，请参阅：
#  1. https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
#  2. https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
#

# AMD GPU 上的 scale 预洗牌
#
# 与 NVIDIA GPU 类似，在具有 CDNA4 架构的 AMD GPU 上，缩放 MFMA 指令
# 原生支持缩放矩阵乘法。由于它仅支持 OCP 微缩放格式，
# 每个 scale 是一个 8 位值，用于缩放 A 或 B 操作数张量中的 32 个元素。
# Scale 以 8 位张量存储。由于 MFMA 指令是 warp 级指令，
# 每个线程向 MFMA 指令提供固定数量的操作数值。
#
# 例如，在形状为 16x16x128 的 MFMA 指令中：
# - 4 个线程沿 K 维贡献元素。
# - 16 个线程沿 M 或 N 维贡献元素。
#
# 从 scale 张量的角度看，即使 K 维在共享内存中连续存储，
# 每个线程看到的 K 维元素也是跨步的（因为与其他线程交织）。
# 这种跨步限制了使用向量化内存访问加载 scale 值的能力。
#
# 我们的目标是重新组织 scale 张量，使得：
# 1. 每个线程将 4 次 MFMA 操作所需的 4 个 scale 值连续存储在内存中。
# 2. 连续线程访问连续内存位置，改善绕过 LDS 时的全局内存合并，
#    这对"细长"矩阵乘法尤为有利。
#
# 我们考虑两种 MFMA 情形：非 K 维为 16 和 32 的情况。
# 两者最小预洗牌分块大小均为 32x32x256。
# 例如，对于 32x256 的操作数分块，对应的 scale 张量形状为 32x8，
# 其中每个 scale 覆盖 K 维上的 32 个元素。
#
# 每个线程为每次 MFMA 操作持有一个 scale。我们将 4 个 scale 值
# （对应 4 次不同的 MFMA 操作）相邻打包存储在内存中。
#
# 情形 1：mfma_scaled_16x16x128
#
# 打包顺序：mfma_op_0, mfma_op_2, mfma_op_1, mfma_op_3
#
#            K = 128       K = 128
#        +------------+ +------------+
#    M=16|  MFMA op 0 | |  MFMA op 1 |
#        +------------+ +------------+
#    M=16|  MFMA op 2 | |  MFMA op 3 |
#        +------------+ +------------+
#
# 情形 2：mfma_scaled_32x32x64
#
# 打包顺序：mfma_op_0, mfma_op_1, mfma_op_2, mfma_op_3
#
#            K=64     K=64     K=64     K=64
#        +--------+ +--------+ +--------+ +--------+
#    M=32| op 0   | | op 1   | | op 2   | | op 3   |
#        +--------+ +--------+ +--------+ +--------+

import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == 'hip' and target.arch == 'gfx950'


def supports_block_scaling():
    return (is_cuda() and torch.cuda.get_device_capability()[0] in [10, 11]) or is_hip_cdna4()


if is_cuda() and torch.cuda.get_device_capability()[0] in [10, 11]:
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_desc,  #
        a_scale_desc,  #
        b_desc,  #
        b_scale_desc,  #
        c_desc,  #
        M: tl.constexpr,  #
        N: tl.constexpr,  #
        K: tl.constexpr,  #
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        rep_m: tl.constexpr,  #
        rep_n: tl.constexpr,  #
        rep_k: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
):  #
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m,
        rep_n,
        rep_k,
        configs["num_stages"],
    )
    return output


def cublas_block_scaled_matmul(a, a_scale, b, b_scale, block_scale_type="mxfp8"):
    """
    cuBLAS 块缩放矩阵乘法基准实现。

    Args:
        a: 输入矩阵 A
            - mxfp8 格式：(M, K)，FP8 E4M3 类型
            - nvfp4 格式：(M, K//2)，uint8 打包 FP4（每字节 2 个元素）
        a_scale: A 的缩放因子
            - mxfp8 格式：E8M0 scales（展平）
            - nvfp4 格式：FP8 E4M3 scales，cublas 布局 (M, K//16)
        b: 输入矩阵 B
            - mxfp8 格式：(N, K)，FP8 E4M3 类型
            - nvfp4 格式：(N, K//2)，uint8 打包 FP4（每字节 2 个元素）
        b_scale: B 的缩放因子
            - mxfp8 格式：E8M0 scales（展平）
            - nvfp4 格式：FP8 E4M3 scales，cublas 布局 (N, K//16)
        block_scale_type: 格式类型（"mxfp8" 或 "nvfp4"）

    Returns:
        output: 结果矩阵 (M, N)，FP16 类型
    """
    M, K_a = a.shape
    N, K_b = b.shape

    if block_scale_type == "mxfp8":
        assert K_a == K_b, "K dimensions must match"
        assert a.dtype == torch.float8_e4m3fn, "Only FP8 E4M3 inputs supported for mxfp8"
        assert b.dtype == torch.float8_e4m3fn, "Only FP8 E4M3 inputs supported for mxfp8"
        # MXFP8 cuBLAS 输出 FP16
        output = torch.empty((M, N), dtype=torch.float16, device="cuda")
        cublas.block_scaled_matmul_mxfp8(a, b, output, a_scale, b_scale)
    elif block_scale_type == "nvfp4":
        # 对于打包 FP4，K_a 和 K_b 以字节为单位（K = K_a * 2 个元素）
        assert K_a == K_b, "K dimensions must match"
        assert a.dtype == torch.uint8, "Only uint8 packed FP4 inputs supported for nvfp4"
        assert b.dtype == torch.uint8, "Only uint8 packed FP4 inputs supported for nvfp4"
        # NVFP4 cuBLAS 输出 FP16
        output = torch.empty((M, N), dtype=torch.float16, device="cuda")
        cublas.block_scaled_matmul_nvfp4(a, b, output, a_scale, b_scale)
    else:
        raise ValueError(f"Unsupported block_scale_type: {block_scale_type}")

    return output


def initialize_block_scaled(M, N, K, block_scale_type="nvfp4", compute_reference=False):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8", "mixed"], f"Invalid block scale type: {block_scale_type}"
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = "cuda"
    a_ref = MXFP4Tensor(size=(M, K), device=device).random()
    # 与 Hopper 的 wgmma 对称 fp8 指令类似，当使用 fp4 操作数时，
    # Blackwell 的 tcgen05.mma 期望 RHS 为列主序布局。
    # 为符合 tl.dot_scaled (M, K) x (K, N) 的预期语义，
    # 数据以列主序生成，沿 K 维打包 fp4，然后逻辑转置。
    # 注意：若某操作数为 fp8 精度，与 Hopper 不同，Blackwell 同时支持
    # RHS 矩阵的行主序和列主序布局。对于混合精度情形，fp4 的 RHS
    # 可以是行或列主序布局，但出于性能考虑，推荐使用列主序布局。
    # 若在混合精度 dot 中为 fp4 RHS 使用 TMA 加载（如本教程），
    # 则必须使用列主序布局。
    b_ref = MXFP4Tensor(size=(N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        # 沿 K 维将两个 fp4 元素打包为一个字节
        a = a_ref.to_packed_tensor(dim=1)

    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=1)

    b_ref = b_ref.to(torch.float32).T

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]
    epsilon = 1e-8
    a_scale = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale = torch.rand(b_scale_shape, device=device) + epsilon

    # 在任何布局转换之前，为 cublas nvfp4 保存原始 scale。
    # 对于 cublas nvfp4，scale 采用原始 4D 布局。
    a_scale_orig = a_scale.clone()
    b_scale_orig = b_scale.clone()

    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    # 使用 5D TMA descriptor [1, rep_m, rep_k, 2, 256]，元素类型为 uint8。
    # 使用 256 个元素可以更好地利用 L2，并避免 TMA 引擎像 32x16xu8 那样
    # 发出大量小消息（16B）。
    a_scale_block_shape = [1, rep_m, rep_k, 2, 256]
    b_scale_block_shape = [1, rep_n, rep_k, 2, 256]
    a_scale = a_scale.reshape(1, a_scale_shape[0], a_scale.shape[1], 2, 256)
    b_scale = b_scale.reshape(1, b_scale_shape[0], b_scale.shape[1], 2, 256)
    a_scale_desc = TensorDescriptor.from_tensor(a_scale, block_shape=a_scale_block_shape)
    b_scale_desc = TensorDescriptor.from_tensor(b_scale, block_shape=b_scale_block_shape)

    reference = None
    if compute_reference:
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale(packed):
            packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
            num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

        a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }

    # 为 cuBLAS 展平 scale
    if block_scale_type == "mxfp8":
        a_scale_cublas = a_scale.contiguous().flatten()
        b_scale_cublas = b_scale.contiguous().flatten()
    elif block_scale_type == "nvfp4":
        a_scale_orig = a_scale_orig.to(torch.float8_e4m3fn)
        b_scale_orig = b_scale_orig.to(torch.float8_e4m3fn)
        a_scale_cublas = a_scale_orig.contiguous().flatten()
        b_scale_cublas = b_scale_orig.contiguous().flatten()

    return a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference, a, b, a_scale_cublas, b_scale_cublas


def validate_block_scaled(M, N, K, block_scale_type="nvfp4"):
    results = initialize_block_scaled(M, N, K, block_scale_type, compute_reference=True)
    a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference = results[:9]
    a, b, a_scale_cublas, b_scale_cublas = results[9:]

    # 测试 Triton 实现
    output = block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, M, N, K, rep_m, rep_n,
                                 rep_k, configs)
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)

    # 测试 cuBLAS 实现（如可用，截至 13.1 仅支持 mxfp8 和 nvfp4）
    if cublas and block_scale_type in ["mxfp8", "nvfp4"]:
        cublas_output = cublas_block_scaled_matmul(a, a_scale_cublas, b, b_scale_cublas,
                                                   block_scale_type=block_scale_type)
        torch.testing.assert_close(reference, cublas_output.to(torch.float32), atol=1e-3, rtol=1e-3)
        print(f"✅ (pass {block_scale_type} - Triton and cuBLAS)")
    else:
        print(f"✅ (pass {block_scale_type} - Triton only)")


def bench_block_scaled(K, block_scale_type="nvfp4", reps=10, warmup_reps=10):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    results = initialize_block_scaled(M, N, K, block_scale_type, compute_reference=False)
    a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, _ = results[:9]
    a, b, a_scale_cublas, b_scale_cublas = results[9:]

    # 预热
    for _ in range(warmup_reps):
        _ = block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, M, N, K, rep_m, rep_n, rep_k,
                                configs)
        if cublas is not None and supports_block_scaling() and block_scale_type in ["mxfp8", "nvfp4"]:
            _ = cublas_block_scaled_matmul(a, a_scale_cublas, b, b_scale_cublas, block_scale_type=block_scale_type)

    # 基准测试
    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, torch.float16, M, N, K, rep_m, rep_n, rep_k,
                                configs)
        if cublas is not None and supports_block_scaling() and block_scale_type in ["mxfp8", "nvfp4"]:
            bytes_per_elem = a.element_size()
            # 对于 nvfp4，K 以元素为单位，但 a.shape[1] 以字节为单位，因此字节计算使用 K/2
            K_bytes = K if block_scale_type == "mxfp8" else K // 2
            with proton.scope(f"cublas [M={M}, N={N}, K={K}]",
                              {"bytes": bytes_per_elem * (M * K_bytes + N * K_bytes + M * N), "flops": 2. * M * N * K}):
                _ = cublas_block_scaled_matmul(a, a_scale_cublas, b, b_scale_cublas, block_scale_type=block_scale_type)
    proton.deactivate(0)
    print("Done benchmarking")


def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer

    metric_names = ["time/ms"]
    metric_names = ["tflop/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


@triton.jit
def block_scaled_matmul_kernel_cdna4(a_ptr, b_ptr, c_ptr, a_scales_ptr, b_scales_ptr, M, N, K, stride_am, stride_ak,
                                     stride_bk, stride_bn, stride_ck, stride_cm, stride_cn, stride_asm, stride_ask,
                                     stride_bsn, stride_bsk,
                                     # 元参数
                                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                     mfma_nonkdim: tl.constexpr):
    """计算矩阵乘法 C = A x B 的 kernel。
    A 和 B 输入采用微缩放 fp4（mxfp4）格式。
    A_scales 和 B_scales 采用 e8m0 格式。
    A 的形状为 (M, K)，B 的形状为 (K, N)，C 的形状为 (M, N)
    """

    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 假设 K 轴上每 32 个元素共享同一个 scale。
    SCALE_GROUP_SIZE: tl.constexpr = 32
    num_k_iter = tl.cdiv(K, BLOCK_K // 2)
    # 为 A 和 B 输入矩阵的第一个 block 创建指针
    # BLOCK 大小以元素为单位，fp4 中每个 uint8 容器打包 2 个元素。
    offs_k = tl.arange(0, BLOCK_K // 2)
    offs_k_split = offs_k
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 为 A 和 B scale 的第一个 block 创建指针
    offs_asn = (pid_n * (BLOCK_N // 32) + tl.arange(0, (BLOCK_N // 32))) % N
    offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE * 32)

    # B scale 的形状为 N x K，尽管 B 操作数的形状为 K x N。
    b_scale_ptrs = (b_scales_ptr + offs_asn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk)
    offs_asm = (pid_m * (BLOCK_M // 32) + tl.arange(0, (BLOCK_M // 32))) % M
    a_scale_ptrs = (a_scales_ptr + offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, num_k_iter):
        # 此处"撤销"全局内存中完成的洗牌操作（shuffle_scales_cdna4 函数）。
        if mfma_nonkdim == 32:
            a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                     1).permute(0, 3, 1, 4, 2,
                                                                5).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 2, 32, 4,
                                                     1).permute(0, 3, 1, 4, 2,
                                                                5).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)
        elif mfma_nonkdim == 16:
            a_scales = tl.load(a_scale_ptrs).reshape(BLOCK_M // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                     1).permute(0, 5, 3, 1, 4, 2,
                                                                6).reshape(BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE)
            b_scales = tl.load(b_scale_ptrs).reshape(BLOCK_N // 32, BLOCK_K // SCALE_GROUP_SIZE // 8, 4, 16, 2, 2,
                                                     1).permute(0, 5, 3, 1, 4, 2,
                                                                6).reshape(BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE)

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs, cache_modifier=None)

        accumulator += tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1")

        # 将指针推进到下一个 K block。
        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk

        a_scale_ptrs += BLOCK_K * stride_ask
        b_scale_ptrs += BLOCK_K * stride_bsk

    c = accumulator.to(c_ptr.type.element_ty)

    # 带掩码地将输出矩阵 C 的 block 写回。
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptrs = (c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask, cache_modifier=".wt")


def shuffle_scales_cdna4(scales: torch.Tensor, mfma_nonkdim: int):
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape

    if mfma_nonkdim == 32:
        scales_shuffled = scales_shuffled.view(sm // 32, 32, sn // 8, 4, 2, 1)
        scales_shuffled = scales_shuffled.permute(0, 2, 4, 1, 3, 5).contiguous()
    elif mfma_nonkdim == 16:
        scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()

    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled


def initialize_block_scaled_amd(M, N, K, mfma_nonkdim):

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    configs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "num_stages": 2,
        "num_warps": 8,
        "mfma_nonkdim": mfma_nonkdim,
    }

    torch.manual_seed(5)

    x = MXFP4Tensor(size=(M, K), device="cuda").random()
    w = MXFP4Tensor(size=(N, K), device="cuda").random()

    x_scales = torch.randint(124, 128, (K // 32, M), dtype=torch.uint8, device="cuda")
    w_scales = torch.randint(124, 128, (K // 32, N), dtype=torch.uint8, device="cuda")
    x_scales = x_scales.T
    w_scales = w_scales.T
    x_scales_shuffled = shuffle_scales_cdna4(x_scales, configs["mfma_nonkdim"])
    w_scales_shuffled = shuffle_scales_cdna4(w_scales, configs["mfma_nonkdim"])

    return (
        x,
        w,
        x_scales,
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        configs,
    )


def validate_block_scaled_amd(M, N, K, block_scale_type="mxfp4", mfma_nonkdim=16):

    def e8m0_to_f32(x):
        x_f32 = 2**((x - 127).to(torch.float32))
        x_f32[x_f32 == 128] = float("nan")
        return x_f32

    def run_torch(x, w, x_scales, w_scales, dtype):
        # 首先将 x 和 w 输入转换为 f32。
        x_f32 = x.to(torch.float32)
        w_f32 = w.to(torch.float32)
        # 再将 e8m0 scale 转换为 f32。
        x_scales = x_scales.repeat_interleave(32, dim=1).to(torch.float32)
        x_scales_f32 = e8m0_to_f32(x_scales)
        x_f32 = x_f32 * x_scales_f32
        w_scales = w_scales.repeat_interleave(32, dim=1).to(torch.float32)
        w_scales_f32 = e8m0_to_f32(w_scales)
        w_f32 = w_f32 * w_scales_f32
        return torch.mm(x_f32, w_f32.T).to(dtype)

    x_mxfp4, w_mxfp4, x_scales, w_scales, x_scales_triton, w_scales_triton, configs = \
    initialize_block_scaled_amd(M, N, K, mfma_nonkdim)

    x = x_mxfp4.to_packed_tensor(dim=1)
    w = w_mxfp4.to_packed_tensor(dim=1)

    triton_out = torch.empty((M, N), device=x.device)
    triton_out = block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs)
    triton_out = triton_out.to(torch.float32)

    torch_out = run_torch(x_mxfp4, w_mxfp4, x_scales, w_scales, torch.float32)
    torch.testing.assert_close(torch_out, triton_out)
    print(f"✅ (pass {block_scale_type}, mfma_nonk_dim {mfma_nonkdim})")


def block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs):
    M, K = x.shape
    N, K = w.shape
    w = w.T
    triton_out = torch.empty((M, N), device=x.device)

    kernel_kwargs = {}
    kernel_kwargs["matrix_instr_nonkdim"] = configs["mfma_nonkdim"]

    BLOCK_M = configs["BLOCK_M"]
    BLOCK_N = configs["BLOCK_N"]

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    triton_out = torch.empty((M, N), device="cuda")

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel_cdna4[grid](x, w, triton_out, x_scales_triton, w_scales_triton, M, N, K, x.stride(0),
                                           x.stride(1), w.stride(0), w.stride(1), 0, triton_out.stride(0),
                                           triton_out.stride(1), x_scales_triton.stride(0), x_scales_triton.stride(1),
                                           w_scales_triton.stride(0), w_scales_triton.stride(1), BLOCK_M, BLOCK_N,
                                           configs["BLOCK_K"], configs["mfma_nonkdim"], num_warps=configs["num_warps"],
                                           num_stages=configs["num_stages"], **kernel_kwargs)
    triton_out = triton_out.to(torch.float32)

    return triton_out


def bench_block_scaled_amd(K, block_scale_type="mxfp4", reps=10, mfma_nonkdim=16):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    x_mxfp4, w_mxfp4, x_scales, w_scales, x_scales_triton, w_scales_triton, configs = \
    initialize_block_scaled_amd(M, N, K, mfma_nonkdim)

    x = x_mxfp4.to_packed_tensor(dim=1)
    w = w_mxfp4.to_packed_tensor(dim=1)

    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul_amd(x, w, x_scales_triton, w_scales_triton, configs)
    proton.deactivate(0)
    print("Done benchmarking")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--bench", action="store_true", default=True)
    parser.add_argument("--format", type=str, choices=["mxfp4", "nvfp4", "mxfp8", "mixed"], default="nvfp4")
    args = parser.parse_args()

    if not supports_block_scaling():
        print("⛔ This example requires GPU support for block scaled matmul")
    else:
        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # 只要不为 0 即可

        torch.manual_seed(42)

        if is_cuda():
            validate_block_scaled(8192, 8192, 8192, block_scale_type=args.format)
        elif is_hip_cdna4():
            assert args.format == "mxfp4", "AMD tutorial only supports mxpf4 format currently"
            validate_block_scaled_amd(8192, 8192, 8192, block_scale_type=args.format, mfma_nonkdim=16)
            validate_block_scaled_amd(8192, 8192, 8192, block_scale_type=args.format, mfma_nonkdim=32)

        if args.bench:
            proton.start("block_scaled_matmul", hook="triton")
            proton.deactivate(0)  # 跳过参数创建阶段
            for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
                if is_cuda():
                    bench_block_scaled(K, reps=10000, block_scale_type=args.format)
                elif is_hip_cdna4():
                    bench_block_scaled_amd(K, reps=10000, block_scale_type=args.format, mfma_nonkdim=16)
                    bench_block_scaled_amd(K, reps=10000, block_scale_type=args.format, mfma_nonkdim=32)
            proton.finalize()
            show_profile("block_scaled_matmul")
