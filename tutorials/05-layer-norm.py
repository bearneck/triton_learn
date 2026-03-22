"""
Layer Normalization（层归一化）
=============================
在本教程中，你将编写一个高性能的 Layer Norm kernel，
其运行速度将超过 PyTorch 的原生实现。

在此过程中，你将学习：

* 在 Triton 中实现反向传播。

* 在 Triton 中实现并行归约（parallel reduction）。

"""

# %%
# 动机
# ----
#
# *LayerNorm* 算子最早在 [BA2016]_ 中提出，用于提升序列模型（如 Transformer）
# 或小 batch 尺寸神经网络的性能。
# 它接受向量 :math:`x` 作为输入，并产生与输入形状相同的向量 :math:`y`。
# 归一化通过对 :math:`x` 减去均值、再除以标准差来完成。
# 归一化之后，再应用由权重 :math:`w` 和偏置 :math:`b` 构成的可学习线性变换。
# 前向传播的表达式如下：
#
# .. math::
#    y = \frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} } * w + b
#
# 其中 :math:`\epsilon` 是加在分母上的小常数，用于保证数值稳定性。
# 下面先来看前向传播的实现。

import torch

import triton
import triton.language as tl

try:
    # 此处引用的是 https://github.com/NVIDIA/apex，而非 PyPI 上的 apex，
    # 因此不应将其添加到 setup.py 的 extras_require 中。
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _layer_norm_fwd_fused(
    X,  # 指向输入的指针
    Y,  # 指向输出的指针
    W,  # 指向权重的指针
    B,  # 指向偏置的指针
    Mean,  # 指向均值的指针
    Rstd,  # 指向 1/std 的指针
    stride,  # 行指针每移动 1 行时的步长
    N,  # X 的列数
    eps,  # 防止除以零的 epsilon
    BLOCK_SIZE: tl.constexpr,
):
    # 将 program id 映射到它需要计算的 X 和 Y 的行。
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # 计算均值
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # 计算方差
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # 写回均值和 rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # 归一化并应用线性变换
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # 写入输出
        tl.store(Y + cols, y, mask=mask)


# %%
# 反向传播
# --------
#
# Layer Norm 算子的反向传播比前向传播稍复杂。
# 设 :math:`\hat{x}` 为线性变换前的归一化输入
# :math:`\frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} }`，
# 则 :math:`x` 的 Vector-Jacobian Product（VJP）:math:`\nabla_{x}` 为：
#
# .. math::
#    \nabla_{x} = \frac{1}{\sigma}\Big( \nabla_{y} \odot w - \underbrace{ \big( \frac{1}{N} \hat{x} \cdot (\nabla_{y} \odot w) \big) }_{c_1} \odot \hat{x} - \underbrace{ \frac{1}{N} \nabla_{y} \cdot w }_{c_2} \Big)
#
# 其中 :math:`\odot` 表示逐元素乘法，:math:`\cdot` 表示点积，:math:`\sigma` 是标准差。
# :math:`c_1` 和 :math:`c_2` 是中间常数，用于提升后续实现的可读性。
#
# 对于权重 :math:`w` 和偏置 :math:`b`，其 VJP :math:`\nabla_{w}` 和 :math:`\nabla_{b}` 更为直接：
#
# .. math::
#    \nabla_{w} = \nabla_{y} \odot \hat{x} \quad \text{and} \quad \nabla_{b} = \nabla_{y}
#
# 由于同一 batch 中所有行共享相同的权重 :math:`w` 和偏置 :math:`b`，其梯度需要求和。
# 为高效完成这一步，我们使用并行归约策略：每个 kernel 实例将若干行的
# 部分 :math:`\nabla_{w}` 和 :math:`\nabla_{b}` 累积到 :math:`\text{GROUP_SIZE_M}` 个独立缓冲区之一中。
# 这些缓冲区驻留在 L2 缓存中，随后由另一个函数进一步归约以计算最终的
# :math:`\nabla_{w}` 和 :math:`\nabla_{b}`。
#
# 设输入行数 :math:`M = 4`，:math:`\text{GROUP_SIZE_M} = 2`，
# 下图展示了 :math:`\nabla_{w}` 的并行归约策略（:math:`\nabla_{b}` 省略）：
#
#   .. image:: parallel_reduction.png
#
# 在 Stage 1 中，颜色相同的 X 行共享同一缓冲区，因此使用锁（lock）来确保
# 每次只有一个 kernel 实例写入该缓冲区。
# 在 Stage 2 中，对缓冲区进一步归约以计算最终的 :math:`\nabla_{w}` 和 :math:`\nabla_{b}`。
# 在下面的实现中，Stage 1 由函数 :code:`_layer_norm_bwd_dx_fused` 实现，
# Stage 2 由函数 :code:`_layer_norm_bwd_dwdb` 实现。


@triton.jit
def _layer_norm_bwd_dx_fused(DX,  # 指向输入梯度的指针
                             DY,  # 指向输出梯度的指针
                             DW,  # 指向权重梯度部分和的指针
                             DB,  # 指向偏置梯度部分和的指针
                             X,  # 指向输入的指针
                             W,  # 指向权重的指针
                             Mean,  # 指向均值的指针
                             Rstd,  # 指向 1/std 的指针
                             Lock,  # 指向锁的指针
                             stride,  # 行指针每移动 1 行时的步长
                             N,  # X 的列数
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # 将 program id 映射到它需要计算的 X、DX 和 DY 的元素。
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # 为并行归约偏移锁指针和权重/偏置梯度指针
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # 将数据加载到 SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # 计算 dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # 写入 dx
    tl.store(DX + cols, dx, mask=mask)
    # 累积 dw/db 的部分和
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # 第一次存储不累积
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # 需要一个屏障以确保所有线程完成后再释放锁
    tl.debug_barrier()

    # 释放锁
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # 指向权重梯度部分和的指针
                         DB,  # 指向偏置梯度部分和的指针
                         FINAL_DW,  # 指向权重梯度的指针
                         FINAL_DB,  # 指向偏置梯度的指针
                         M,  # GROUP_SIZE_M
                         N,  # 列数
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # 将 program id 映射到它需要计算的 DW 和 DB 的元素。
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 遍历 DW 和 DB 的各行以求和部分和。
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # 将最终求和结果写入输出。
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


# %%
# 基准测试
# --------
#
# 现在可以将我们的 kernel 与 PyTorch 的性能进行对比。
# 此处我们聚焦于每个特征维度小于 64KB 的输入。
# 具体地，可将 :code:`'mode': 'backward'` 设置为测试反向传播。


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # 分配输出
        y = torch.empty_like(x)
        # 将输入数据重塑为 2D 张量
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # 每个特征维度小于 64KB：启动融合 kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # 启发式确定 warp 数量
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # 启动 kernel
        _layer_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # 启发式确定 DW/DB 的并行归约流数量
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # 分配输出
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # 使用前向传播的启发式参数启动 kernel
        # 同时计算 DW 和 DB 的部分和
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M, )](  #
            dx, dy, _dw, _db, x, w, m, v, locks,  #
            x_arg.stride(0), N,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
        # 在独立的 kernel 中累积部分和
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply


def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # 创建数据
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # 前向传播
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # 反向传播（triton）
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # 反向传播（torch）
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # 对比结果
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    ))
def bench_layer_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device=DEVICE):
    # 创建数据
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "apex":
            apex_layer_norm = (apex.normalization.FusedLayerNorm(w_shape).to(x.device).to(x.dtype))
            return apex_layer_norm(x)  # noqa: F811, E704

    # 前向传播
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # 反向传播
    if mode == 'backward':
        y = y_fwd()
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: F811, E704
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


test_layer_norm(1151, 8192, torch.float16)
bench_layer_norm.run(save_path='.', print_data=True)

# %%
# 参考文献
# --------
#
# .. [BA2016] Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton, "Layer Normalization", Arxiv 2016
