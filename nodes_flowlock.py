"""H3 Face Flow Lock: pins a refined face back onto the features of its base frames.

The problem: H3 re-draws the face crop at img2img denoise ~0.3 and puts the features (eyes, brows,
mouth corners) 1-2 px away from where the base had them - and the offset is different on every
frame. Hair, ears and the head outline stay the base's, so the new face slides ("dances") on a
head that does not move with it. Better tracking does not help: the crop is pasted back with the
exact inverse transform; the drift is inside H3's output.

The fix, per frame:
  1. dense optical flow base -> refined (OpenCV DIS) on lightly blurred grey frames, so it follows
     structure, not the new skin texture;
  2. only where the frame was changed (|refined - base|, blurred, > threshold, dilated, feathered);
  3. the flow is smoothed spatially (Gaussian, mask-weighted) - it follows the misplacement of the
     features, never the new detail; no temporal smoothing (the wobble IS frame-to-frame);
  4. the refined frame is warped by that flow (bicubic) and blended through the mask.

Measured (MiniMax H3 music video, 2 people, 107 frames, 3072x1792, face 206 px tall): the
frame-to-frame change of the refined-vs-base feature offset went from 0.73% to 0.19% of the face
height (a whole-frame latent re-render of the same shot: 0.38%); face sharpness (Laplacian
variance) 239 -> 260, i.e. no detail lost.

Put it right after H3 Face Stitch / H3 Face Stitch Plus: base_images = the stitch's base frames,
refined_images = the stitch's output. Needs opencv-python (DIS is in the main package).

Speed (v0.2): the change mask (diff, threshold, dilate, feather) is computed on the GPU for a batch of frames; the
optical flow, flow smoothing and warp then run only inside the box around the changed pixels (the face), on several
CPU threads. On 107 frames at 3072x1792 this took the node from ~120 s to a few seconds; the output is the same
except within a few px of that box's padded edge, where the flow field was already ~0.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

import comfy.utils
import comfy.model_management as mm

try:
    from .gpu_ops import gauss
except ImportError:  # loaded as a plain module (tests)
    from gpu_ops import gauss


def _to_u8(img: torch.Tensor) -> np.ndarray:
    return (img[..., :3].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def flow_lock_frame(base: np.ndarray, ref: np.ndarray, sigma: float, threshold: float, dis, cv2):
    """base, ref: [H,W,3] uint8 RGB. Returns (locked [H,W,3] uint8, p95 correction px or None if unchanged)."""
    H, W = base.shape[:2]
    d = cv2.GaussianBlur(cv2.absdiff(base, ref).max(2).astype(np.float32), (0, 0), 3)
    m = (d > threshold).astype(np.float32)
    if m.sum() < 50:
        return ref, None
    m = cv2.dilate(m, np.ones((15, 15), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    grey = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (0, 0), 1.2)
    f = dis.calc(grey(base), grey(ref), None)  # base(x) ~ ref(x + f)
    w = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-4
    f = np.dstack([cv2.GaussianBlur(f[..., k] * m, (0, 0), sigma) / w for k in range(2)]) * (m[..., None] > 0.02)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    warped = cv2.remap(ref, gx + f[..., 0].astype(np.float32), gy + f[..., 1].astype(np.float32),
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    mk = np.clip(m * 1.5, 0, 1)[..., None]
    out = np.clip(ref * (1 - mk) + warped * mk, 0, 255).astype(np.uint8)
    core = mk[..., 0] > 0.5
    p95 = float(np.percentile(np.hypot(f[..., 0], f[..., 1])[core], 95)) if core.any() else 0.0
    return out, p95


def _change_masks(base: torch.Tensor, ref: torch.Tensor, sigma: float, threshold: float) -> torch.Tensor:
    """base, ref: (N, H, W, 3) 0..1 on the GPU -> feathered change masks (N, H, W) as in flow_lock_frame"""
    d = ((base - ref).abs() * 255.0).amax(-1, keepdim=True).movedim(-1, 1)
    d = gauss(d, 3.0)
    m = (d > threshold).float()
    keep = (m.flatten(1).sum(1) >= 50).float().view(-1, 1, 1, 1)
    m = F.max_pool2d(m, 15, stride=1, padding=7)
    return (gauss(m, sigma) * keep)[:, 0]


def flow_lock_roi(base: np.ndarray, ref: np.ndarray, m: np.ndarray, sigma: float, dis, cv2):
    """base, ref: [h,w,3] uint8 RGB crops around the changed area, m: their feathered change mask (float32).
    Same steps as flow_lock_frame from the flow on. Returns (locked crop, p95)."""
    h, w = base.shape[:2]
    grey = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (0, 0), 1.2)
    f = dis.calc(grey(base), grey(ref), None)
    wt = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-4
    f = np.dstack([cv2.GaussianBlur(f[..., k] * m, (0, 0), sigma) / wt for k in range(2)]) * (m[..., None] > 0.02)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(ref, gx + f[..., 0].astype(np.float32), gy + f[..., 1].astype(np.float32),
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    mk = np.clip(m * 1.5, 0, 1)[..., None]
    out = np.clip(ref * (1 - mk) + warped * mk, 0, 255).astype(np.uint8)
    core = mk[..., 0] > 0.5
    p95 = float(np.percentile(np.hypot(f[..., 0], f[..., 1])[core], 95)) if core.any() else 0.0
    return out, p95


class H3FaceFlowLock:
    """Warp the refined face onto the base's features with smoothed optical flow (kills face wobble)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_images": ("IMAGE", {"tooltip": "The frames the face was pasted onto (the stitch's base_images)."}),
                "refined_images": ("IMAGE", {"tooltip": "The stitch's output, same size as base_images."}),
                "sigma_frac": ("FLOAT", {"default": 1.0 / 150.0, "min": 0.001, "max": 0.05, "step": 0.0005,
                    "tooltip": "Flow smoothing: Gaussian sigma as a fraction of the frame height (1/150 = 12 px "
                               "at 1792). Larger follows only the global offset (1/90 left part of the wobble); "
                               "smaller starts to undo intended local changes."}),
                "threshold": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5,
                    "tooltip": "Minimum |refined - base| (0-255, blurred) that counts as a changed pixel. "
                               "Unchanged pixels are passed through."}),
                "base_offset": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "refined frame k is compared with base frame base_offset + k."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = "Pin the refined face onto the base's features (removes the frame-to-frame face wobble)."

    def run(self, base_images, refined_images, sigma_frac, threshold, base_offset):
        import cv2

        n = refined_images.shape[0]
        if base_images.shape[0] < base_offset + n:
            raise ValueError(f"base_images has {base_images.shape[0]} frames, need base_offset + {n}.")
        if tuple(base_images.shape[1:3]) != tuple(refined_images.shape[1:3]):
            raise ValueError(f"size mismatch: base {tuple(base_images.shape[1:3])} vs refined "
                             f"{tuple(refined_images.shape[1:3])} - flow-lock at the stitch's output size.")
        H = refined_images.shape[1]
        sigma = max(1.0, float(sigma_frac) * H)
        dev = mm.get_torch_device()
        W = refined_images.shape[2]
        out = refined_images[..., :3].clone()
        pad = int(4 * sigma) + 48   # the flow smoothing reaches ~4 sigma; DIS needs some context around the face
        tls = threading.local()

        def one(k, mk):
            mm.throw_exception_if_processing_interrupted()
            ys, xs = np.nonzero(mk > 1e-3)
            if ys.size == 0:
                return k, None, None
            y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + 1 + pad)
            x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + 1 + pad)
            if not hasattr(tls, "dis"):
                tls.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            b = _to_u8(base_images[base_offset + k, y0:y1, x0:x1])
            r = _to_u8(refined_images[k, y0:y1, x0:x1])
            locked, p95 = flow_lock_roi(b, r, np.ascontiguousarray(mk[y0:y1, x0:x1]), sigma, tls.dis, cv2)
            return k, (y0, y1, x0, x1, locked), p95

        mags = []
        pbar = comfy.utils.ProgressBar(n)
        chunk = 8
        with ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 4) - 2))) as ex:
            for c0 in range(0, n, chunk):
                mm.throw_exception_if_processing_interrupted()
                c1 = min(n, c0 + chunk)
                with torch.no_grad():
                    ms = _change_masks(base_images[base_offset + c0:base_offset + c1, ..., :3].to(dev).float(),
                                       refined_images[c0:c1, ..., :3].to(dev).float(), sigma, float(threshold)).cpu().numpy()
                for k, roi, p95 in ex.map(one, range(c0, c1), list(ms)):
                    if roi is not None:
                        y0, y1, x0, x1, locked = roi
                        out[k, y0:y1, x0:x1] = torch.from_numpy(locked.astype(np.float32) / 255.0).to(out.dtype)
                        mags.append(p95)
                    pbar.update(1)
        report = (f"flow lock: {len(mags)}/{n} frame(s) changed and locked, sigma {sigma:.1f}px, "
                  f"correction p95 {np.mean(mags) if mags else 0:.2f}px mean / {max(mags) if mags else 0:.2f}px max")
        print("[H3FaceRefine+] " + report)
        return (out, report)
