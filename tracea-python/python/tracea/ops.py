import torch
import tracea

_CTX = None

def _get_ctx():
    global _CTX
    if _CTX is None:
        _CTX = tracea.Context()
    return _CTX

def gemm(x, w, bias=None, epilogue=None):
    """
    Perform General Matrix Multiplication: C = X @ W
    
    Args:
        x (torch.Tensor): Input tensor (M x K)
        w (torch.Tensor): Weight tensor (K x N)
        bias (torch.Tensor, optional): Bias tensor (N)
        epilogue (str, optional): Epilogue operations (e.g., "bias+relu")
        
    Returns:
        torch.Tensor: Output tensor (M x N)
    """
    ctx = _get_ctx()
    m, k = x.shape
    k_w, n = w.shape
    if k != k_w:
        raise ValueError(f"Shape mismatch: x({m}x{k}) @ w({k_w}x{n})")
    
    # Allocate output
    # TODO: Helper to allocate based on device?
    # For now assuming CUDA.
    c = torch.empty((m, n), dtype=x.dtype, device=x.device)
    
    ctx.gemm(x, w, c, m, n, k, epilogue=epilogue, bias=bias)
    return c

def conv2d(x, w, bias=None, residual=None, epilogue=None, stride=1, padding=0, dilation=1):
    """
    Perform 2D Convolution.
    
    Args:
        x (torch.Tensor): Input (N, H, W, C) - Channels Last
        w (torch.Tensor): Weight (K, R, S, C) - Channels Last
        bias (torch.Tensor, optional): Bias (K)
        
    Returns:
        torch.Tensor: Output (N, H_out, W_out, K)
    """
    ctx = _get_ctx()
    
    # Check dims (assuming NHWC)
    if x.ndim != 4 or w.ndim != 4:
         raise ValueError("Expected NHWC input and KRSC weight")

    n_batch, h, w_in, c = x.shape
    k, r, s, c_w = w.shape
    
    if c != c_w:
        raise ValueError(f"Channel mismatch: x({c}) vs w({c_w})")
        
    # Output shape
    h_out = (h + 2*padding - r) // stride + 1
    w_out = (w_in + 2*padding - s) // stride + 1
    
    o = torch.empty((n_batch, h_out, w_out, k), dtype=x.dtype, device=x.device)
    
    ctx.conv2d(x, w, o, n_batch, c, h, w_in, k, r, s, stride=stride, pad=padding, dilation=dilation, epilogue=epilogue, bias=bias, residual=residual)
    return o

def conv_transpose2d(x, w, bias=None, residual=None, epilogue=None, stride=1, padding=0, output_padding=0, dilation=1):
    """
    Perform 2D Transposed Convolution.
    """
    ctx = _get_ctx()
    n_batch, h, w_in, c = x.shape
    k, r, s, c_w = w.shape # Weight might be (C, R, S, K) or similar depending on layout? 
    # Tracea conv_transpose uses K as output channels.
    
    h_out = (h - 1) * stride - 2 * padding + r + output_padding
    w_out = (w_in - 1) * stride - 2 * padding + s + output_padding
    
    # Alloc output (N, H_out, W_out, K)
    o = torch.empty((n_batch, h_out, w_out, k), dtype=x.dtype, device=x.device)
    
    ctx.conv_transpose2d(x, w, o, n_batch, c, h, w_in, k, r, s, stride=stride, pad=padding, output_padding=output_padding, dilation=dilation, epilogue=epilogue, bias=bias, residual=residual)
    return o

def attention(q, k, v, causal=True, softmax_mode="auto"):
    """
    Perform FlashAttention-2.
    Inputs: (B, H, S, D)
    """
    ctx = _get_ctx()
    b, h, s, d = q.shape
    # Validate shapes...
    
    # Output
    o = torch.empty_like(q)
    
    # Attention needs dh_in? d is dh_in.
    # Sig: (q, k, v, o, b, h, s, d, dh, causal, ...)
    # python.rs: attention(q, k, v, o, b, h, s, d, dh, causal, scale_sqrt, ...)
    
    ctx.attention(q, k, v, o, b, h, s, d, d, causal=causal, softmax_mode=softmax_mode)
    return o
