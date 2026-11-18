"""GPU helpers shared by H3 Face Flow Lock and H3 Face Tone Match (torch, any device).

gauss()   - separable Gaussian blur, same kernel size rule and border (reflect-101) as cv2.GaussianBlur on float32
rgb2lab() / lab2rgb() - sRGB <-> CIE L*a*b* (D65) with the constants OpenCV uses for float images
            (L 0..100, a/b about -127..127), so the corrections keep the units of the earlier CPU version.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _kernel(sigma: float, device, dtype):
    r = max(1, int(round(sigma * 4.0)))  # cv2: ksize = round(sigma * 8 + 1) | 1 for float images
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum(), r


def gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x: (N, C, H, W). Blur over H and W."""
    if sigma <= 0:
        return x
    k, r = _kernel(float(sigma), x.device, x.dtype)
    N, C, H, W = x.shape
    y = x.reshape(N * C, 1, H, W)
    # reflect padding needs pad < size; fall back to replicate on tiny images
    mode = "reflect" if r < min(H, W) else "replicate"
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode=mode), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode=mode), k.view(1, 1, -1, 1))
    return y.reshape(N, C, H, W)


_M = [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
_WX, _WZ = 0.950456, 1.088754


def rgb2lab(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (..., 3) in 0..1 -> (..., 3) L, a, b"""
    c = rgb.clamp(0, 1)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = torch.tensor(_M, device=rgb.device, dtype=rgb.dtype)
    xyz = lin @ M.T
    X, Y, Z = xyz[..., 0] / _WX, xyz[..., 1], xyz[..., 2] / _WZ
    f = lambda t: torch.where(t > 0.008856, t.clamp(min=1e-12) ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f(X), f(Y), f(Z)
    L = torch.where(Y > 0.008856, 116.0 * fy - 16.0, 903.3 * Y)
    return torch.stack([L, 500.0 * (fx - fy), 200.0 * (fy - fz)], -1)


def lab2rgb(lab: torch.Tensor) -> torch.Tensor:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    Y = torch.where(L > 7.9996, fy ** 3, L / 903.3)
    fx, fz = a / 500.0 + fy, fy - b / 200.0
    inv = lambda t: torch.where(t > 0.206893, t ** 3, (t - 16.0 / 116.0) / 7.787)
    xyz = torch.stack([inv(fx) * _WX, Y, inv(fz) * _WZ], -1)
    Mi = torch.linalg.inv(torch.tensor(_M, device=lab.device, dtype=torch.float64)).to(lab.dtype)
    lin = (xyz @ Mi.T).clamp(0, 1)
    return torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.clamp(min=1e-12) ** (1.0 / 2.4) - 0.055).clamp(0, 1)
