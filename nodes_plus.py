"""Quality add-ons for ComfyUI-H3-FaceRefine.

Where the stock pipeline loses face detail, and what these nodes do about it:

1. H3 Face Track Crop magnifies a small face to the canvas with ONE bilinear sample. A 40 px
   face blown up 5-6x reaches H3 as a smooth blob, and at img2img denoise 0.4 the turbo model
   cannot invent the pores, lashes and lip lines that were never in its input.
   -> H3FaceCropUpscale rebuilds the same crops through a super-resolution model (ESRGAN family,
      UPSCALE_MODEL), so H3 starts from a sharp face. H3 then re-denoises it with temporal
      attention, which removes the per-frame flicker SR alone would have.

2. H3 Face Stitch maps the 768 px result back onto a ~100 px box with a single bilinear
   grid_sample: no low-pass before decimation, so detail aliases (shimmer) or is simply skipped.
   -> H3FaceStitchPlus first shrinks the crop with an antialiased resize to the box size, then
      places it with a bicubic sub-pixel sample.

3. Stock colour match is one mean/std per channel over the whole mask, so local light (a shadow
   across a cheek, a rim light) does not follow, and the face reads as pasted on.
   -> freq_sep: low frequencies (light, tone, colour) come from the ORIGINAL frame, high
      frequencies (features, skin texture) from the refined crop: out = blur(base) +
      gain * (refined - blur(refined)). The cutoff scales with face size.

4. A face that is 60 px tall in the delivered frame can never hold more than 60 px of detail.
   -> H3FaceStitchPlus pastes onto base frames of ANY resolution: feed it the clip upscaled 2x
      (SeedVR2 / FlashVSR) and the crop boxes are scaled to it, so the refined face keeps 2x the
      pixels.

Helpers marked "from ComfyUI-H3-FaceRefine" are copied from that MIT-licensed pack
(Copyright (c) 2026 Carasibana) so this pack does not depend on its import path.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _gaussian_blur_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """from ComfyUI-H3-FaceRefine: separable Gaussian on [N,1,H,W], sigma = k/6."""
    if feather <= 0:
        return mask
    k = 2 * int(feather) + 1
    shortest = min(mask.shape[-2], mask.shape[-1])
    if shortest <= k:
        k = max(3, int(shortest / 2) | 1)
    sigma = max(k / 6.0, 0.5)
    x = torch.arange(k, device=mask.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(mask.dtype)
    pad = k // 2
    m = F.conv2d(F.pad(mask, (pad, pad, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    m = F.conv2d(F.pad(m, (0, 0, pad, pad), mode="replicate"), g.view(1, 1, k, 1))
    return m


def _face_region_mask(ch, cw, face_rect, dilation, feather, shape, device):
    """from ComfyUI-H3-FaceRefine: solid face box (or ellipse) in canvas space, dilated, blurred."""
    m = torch.zeros((1, 1, int(ch), int(cw)), device=device, dtype=torch.float32)
    fx, fy, fwd, fhd = face_rect
    fx -= dilation; fy -= dilation
    fwd += 2 * dilation; fhd += 2 * dilation
    if shape == "ellipse":
        yy = torch.arange(ch, device=device, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(cw, device=device, dtype=torch.float32).view(1, -1)
        ccx, ccy = fx + fwd / 2.0, fy + fhd / 2.0
        rx, ry = max(fwd / 2.0, 1.0), max(fhd / 2.0, 1.0)
        m[0, 0] = (((xx - ccx) / rx) ** 2 + ((yy - ccy) / ry) ** 2 <= 1.0).float()
    else:
        x0 = max(0, int(round(fx))); y0 = max(0, int(round(fy)))
        x1 = min(int(cw), int(round(fx + fwd))); y1 = min(int(ch), int(round(fy + fhd)))
        if x1 > x0 and y1 > y0:
            m[0, 0, y0:y1, x0:x1] = 1.0
    return _gaussian_blur_mask(m, feather).clamp(0, 1)


def _feather_mask(h, w, feather, device):
    """from ComfyUI-H3-FaceRefine: 1 in the core, cosine ramp to 0 at every edge."""
    m = torch.ones((h, w), device=device, dtype=torch.float32)
    f = int(max(0, min(feather, min(h, w) // 2 - 1)))
    if f <= 0:
        return m
    ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, np.pi, f + 2, device=device)[1:-1])
    m[:f, :] *= ramp.view(-1, 1)
    m[h - f:, :] *= ramp.flip(0).view(-1, 1)
    m[:, :f] *= ramp.view(1, -1)
    m[:, w - f:] *= ramp.flip(0).view(1, -1)
    return m


def _gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian with a TRUE sigma (px) on [N,C,H,W], replicate edges."""
    if sigma < 0.3:
        return x
    r = int(math.ceil(3.0 * sigma))
    r = max(1, min(r, (min(x.shape[-2], x.shape[-1]) - 1)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    g = torch.exp(-(t ** 2) / (2.0 * sigma ** 2))
    g = (g / g.sum()).to(x.dtype)
    C = x.shape[1]
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return y


def _box_theta(rx0, ry0, rw, rh, x, y, bw, bh):
    """affine_grid theta mapping an output window (rx0,ry0,rw,rh) into the normalised
    coordinates of the float box (x,y,bw,bh). align_corners=False on both sides, so the
    mapping does not depend on the texture's pixel size."""
    return torch.tensor([[[rw / bw, 0.0, (2.0 * rx0 + rw - 2.0 * x) / bw - 1.0],
                          [0.0, rh / bh, (2.0 * ry0 + rh - 2.0 * y) / bh - 1.0]]],
                        dtype=torch.float32)


def _sample_box(tex: torch.Tensor, box, out_w: int, out_h: int, mode="bicubic") -> torch.Tensor:
    """Resample the float box of `tex` [1,C,H,W] to [1,C,out_h,out_w].

    When the box is LARGER than the output, the whole texture is first shrunk with an
    antialiased resize so that the final sub-pixel sample runs at ~1:1 - a plain
    grid_sample would decimate without a low-pass and alias."""
    x, y, bw, bh = box
    H, W = tex.shape[-2:]
    r = min(out_w / bw, out_h / bh)
    if r < 0.95:
        nw, nh = max(2, int(round(W * r))), max(2, int(round(H * r)))
        sx, sy = nw / W, nh / H
        tex = F.interpolate(tex, size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
        x, y, bw, bh = x * sx, y * sy, bw * sx, bh * sy
        H, W = nh, nw
    th = torch.tensor([[[bw / W, 0.0, (2.0 * x + bw) / W - 1.0],
                        [0.0, bh / H, (2.0 * y + bh) / H - 1.0]]], dtype=torch.float32, device=tex.device)
    grid = F.affine_grid(th, (1, tex.shape[1], int(out_h), int(out_w)), align_corners=False)
    return F.grid_sample(tex, grid, mode=mode, padding_mode="border", align_corners=False)


# ----------------------------------------------------------------------------
# 1. super-resolved crops
# ----------------------------------------------------------------------------

class H3FaceCropUpscale:
    """Rebuild H3 Face Track Crop's crops through a super-resolution model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The SAME frames that went into H3 Face Track Crop."}),
                "transform": ("H3FACEXFORM",),
                "upscale_model": ("UPSCALE_MODEL", {
                    "tooltip": "An ESRGAN-family model from Load Upscale Model. 4x-UltraSharp keeps "
                               "texture; 4x-ClearRealityV1 is cleaner on compressed video."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Mix between the plain bicubic crop (0) and the SR crop (1)."}),
                "min_magnification": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Frames magnified less than this skip SR (the face is already big "
                               "enough; only an antialiased resample is done)."}),
                "second_pass_above": ("FLOAT", {"default": 6.0, "min": 2.0, "max": 32.0, "step": 0.5,
                    "tooltip": "Magnification above which the SR model runs twice (4x -> 16x) "
                               "instead of finishing with a bicubic stretch. Tiny faces only."}),
                "tile": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("crops", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = "Super-resolved face crops for H3 (replaces Track Crop's bilinear crops)."

    def _sr(self, model, x, tile, dev):
        """x [1,3,h,w] on dev -> SR result [1,3,h*s,w*s]."""
        s = float(getattr(model, "scale", 4))
        if max(x.shape[-2:]) <= tile:
            return model(x).clamp(0, 1)
        return comfy.utils.tiled_scale(x, lambda a: model(a), tile_x=tile, tile_y=tile,
                                       overlap=32, upscale_amount=s, output_device=dev).clamp(0, 1)

    def run(self, images, transform, upscale_model, strength, min_magnification,
            second_pass_above, tile):
        dev = mm.get_torch_device()
        boxes = transform["boxes"]
        cw, ch = transform["canvas"]
        W, H = transform["src_size"]
        source = transform.get("source") or list(range(len(boxes)))
        if images.shape[2] != W or images.shape[1] != H:
            raise ValueError(f"images are {images.shape[2]}x{images.shape[1]} but the transform was "
                             f"built on {W}x{H}: feed the same frames that went into H3 Face Track Crop.")

        need = mm.module_size(upscale_model.model) + 512 * 512 * 3 * 4 * 16 * 384
        mm.free_memory(need, dev)
        upscale_model.to(dev)

        K = len(boxes)
        out = torch.zeros((K, ch, cw, 3), dtype=images.dtype)
        mags, n_sr, n_two = [], 0, 0
        pbar = comfy.utils.ProgressBar(K)
        try:
            for k in range(K):
                mm.throw_exception_if_processing_interrupted()
                i = int(source[k])
                x, y, bw, bh = (float(v) for v in boxes[k])
                mag = ch / bh
                mags.append(mag)
                pad = 6
                x0, y0 = max(0, int(math.floor(x)) - pad), max(0, int(math.floor(y)) - pad)
                x1, y1 = min(W, int(math.ceil(x + bw)) + pad), min(H, int(math.ceil(y + bh)) + pad)
                patch = images[i:i + 1, y0:y1, x0:x1, :3].to(dev).movedim(-1, 1).float()
                lbox = (x - x0, y - y0, bw, bh)
                plain = _sample_box(patch, lbox, cw, ch)
                if mag < min_magnification or strength <= 0:
                    res = plain
                else:
                    up = self._sr(upscale_model, patch, tile, dev)
                    n_sr += 1
                    if mag > second_pass_above:
                        up = self._sr(upscale_model, up, tile, dev)
                        n_two += 1
                    sx, sy = up.shape[-1] / patch.shape[-1], up.shape[-2] / patch.shape[-2]
                    sr = _sample_box(up, (lbox[0] * sx, lbox[1] * sy, bw * sx, bh * sy), cw, ch)
                    res = plain + (sr - plain) * float(strength)
                out[k] = res.clamp(0, 1)[0].movedim(0, -1).to("cpu", out.dtype)
                pbar.update(1)
        finally:
            upscale_model.to("cpu")

        m = np.array(mags) if mags else np.zeros(1)
        report = (f"crop upscale: {K} crops {cw}x{ch}, magnification {m.min():.2f}-{m.max():.2f}x "
                  f"(median {np.median(m):.2f}); SR on {n_sr} frame(s), double SR on {n_two}; "
                  f"strength {strength:.2f}")
        print("[H3FaceRefine+] " + report)
        return (out, report)


# ----------------------------------------------------------------------------
# 2. stitch: antialiased, frequency-separated, any base resolution
# ----------------------------------------------------------------------------

class H3FaceStitchPlus:
    """H3 Face Stitch with antialiased placement, frequency-separated tone transfer and
    support for base frames at a higher resolution than the tracked clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_images": ("IMAGE", {
                    "tooltip": "Frames to paste onto. Same resolution as the tracked clip, OR the "
                               "same clip upscaled (e.g. 2x with SeedVR2): the boxes are scaled to "
                               "it automatically and the face keeps the extra pixels."}),
                "refined_crops": ("IMAGE",),
                "transform": ("H3FACEXFORM",),
                "paste_region": (["face_only", "face_ellipse", "full_crop"], {"default": "face_only"}),
                "mask_dilation": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2,
                    "tooltip": "Grow the paste mask, in CANVAS px (as H3 Face Stitch)."}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 256, "step": 2,
                    "tooltip": "Mask blur in SOURCE px of the tracked clip (as H3 Face Stitch, "
                               "scaled along with the base)."}),
                "tone_mode": (["freq_sep", "mean_std", "none"], {"default": "freq_sep",
                    "tooltip": "freq_sep: light/colour from the original frame, detail from the "
                               "refined face (follows local shadows, no tint shift). mean_std: the "
                               "stock global colour match. none: paste as rendered."}),
                "lowpass_face_frac": ("FLOAT", {"default": 0.15, "min": 0.02, "max": 1.0, "step": 0.01,
                    "tooltip": "freq_sep cutoff: Gaussian sigma as a fraction of the face height. "
                               "Lower keeps more of the original (only fine detail is replaced); "
                               "higher lets the refined face change larger shapes. 0.10-0.25."}),
                "detail_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "freq_sep only: multiplier on the refined high frequencies. 1.0 = as "
                               "rendered; 1.1-1.3 gives back the crispness lost to the VAE."}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "undetected_frames": (["fade_out", "skip", "composite_anyway"], {"default": "fade_out"}),
            },
            "optional": {
                "masks": ("MASK", {"tooltip": "Per-frame canvas masks (H3 Face Mask SAM). "
                                              "Overrides paste_region; use a small feather (4-8)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = "Composite refined face crops back: antialiased, frequency-separated, any base size."

    def run(self, base_images, refined_crops, transform, paste_region, mask_dilation, feather,
            tone_mode, lowpass_face_frac, detail_gain, blend, undetected_frames="fade_out", masks=None):
        dev = mm.get_torch_device()
        boxes = transform["boxes"]
        cw, ch = transform["canvas"]
        W, H = transform["src_size"]
        face_rects = transform.get("face_rect")
        BW, BH = base_images.shape[2], base_images.shape[1]
        S = BW / float(W)
        if abs(BH / float(H) - S) > 0.02 * S:
            raise ValueError(f"base_images {BW}x{BH} are not a uniform rescale of the tracked "
                             f"clip {W}x{H}.")

        if undetected_frames == "composite_anyway":
            weights = None
        elif undetected_frames == "skip":
            weights = [1.0 if d else 0.0 for d in transform.get("detected", [])] or None
        else:
            weights = transform.get("weights")
        B = min(len(boxes), refined_crops.shape[0])
        source = transform.get("source")
        if not source or len(source) < B:
            source = list(range(min(B, base_images.shape[0])))
            B = len(source)

        out = base_images[..., :3].clone()
        dt = out.dtype
        sig_list, mag_list = [], []
        pbar = comfy.utils.ProgressBar(B)

        for k in range(B):
            mm.throw_exception_if_processing_interrupted()
            w = float(blend) * (float(weights[k]) if weights is not None and k < len(weights) else 1.0)
            if w <= 1e-4:
                pbar.update(1)
                continue
            f = int(source[k])
            xs, ys, bws, bhs = (float(v) for v in boxes[k])
            x, y, bw, bh = xs * S, ys * S, bws * S, bhs * S

            # ---- paste mask in canvas space (identical construction to H3 Face Stitch)
            f_can = max(1, min(int(round(feather * (ch / max(bhs, 1.0)))), ch // 3))
            if masks is not None:
                mk = masks[k:k + 1].to(dev).float().unsqueeze(1)
                if mk.shape[-2:] != (ch, cw):
                    mk = F.interpolate(mk, size=(ch, cw), mode="bilinear", align_corners=False)
                if mask_dilation > 0:
                    kk = 2 * int(mask_dilation) + 1
                    mk = F.max_pool2d(mk, kk, stride=1, padding=kk // 2)
                mask_can = _gaussian_blur_mask(mk, f_can).clamp(0, 1)
            elif paste_region == "full_crop":
                mask_can = _feather_mask(ch, cw, f_can, dev).view(1, 1, ch, cw)
            else:
                fr = (face_rects[k] if face_rects and k < len(face_rects)
                      else (cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5))
                mask_can = _face_region_mask(ch, cw, fr, int(mask_dilation), f_can,
                                             "ellipse" if paste_region == "face_ellipse" else "rect", dev)

            # face height in OUTPUT px -> frequency-separation sigma
            fh_can = float(face_rects[k][3]) if face_rects and k < len(face_rects) else ch / 2.5
            face_h = fh_can / ch * bh
            sigma = max(0.8, float(lowpass_face_frac) * face_h)
            sig_list.append(sigma)
            mag_list.append(ch / bh)

            # ---- output window: the box, plus room for the low-pass, clipped to the frame
            mrg = int(math.ceil(3 * sigma)) + 4 if tone_mode == "freq_sep" else 2
            rx0, ry0 = max(0, int(math.floor(x)) - mrg), max(0, int(math.floor(y)) - mrg)
            rx1, ry1 = min(BW, int(math.ceil(x + bw)) + mrg), min(BH, int(math.ceil(y + bh)) + mrg)
            rw, rh = rx1 - rx0, ry1 - ry0
            if rw < 2 or rh < 2:
                pbar.update(1)
                continue

            # ---- antialiased shrink of crop + mask to the box size, then 1:1 bicubic placement
            tex = refined_crops[k:k + 1, ..., :3].to(dev).movedim(-1, 1).float()
            mtex = mask_can
            tw, tht = int(round(bw)), int(round(bh))
            if tht < ch * 0.95:
                tw, tht = max(4, tw), max(4, tht)
                tex = F.interpolate(tex, size=(tht, tw), mode="bilinear", antialias=True, align_corners=False)
                mtex = F.interpolate(mtex, size=(tht, tw), mode="bilinear", antialias=True, align_corners=False)
            th = _box_theta(rx0, ry0, rw, rh, x, y, bw, bh).to(dev)
            grid = F.affine_grid(th, (1, 3, rh, rw), align_corners=False)
            patch = F.grid_sample(tex, grid, mode="bicubic", padding_mode="border", align_corners=False)
            m = F.grid_sample(mtex, grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0, 1)

            base = out[f, ry0:ry1, rx0:rx1, :].to(dev).float().movedim(-1, 0).unsqueeze(0)

            if tone_mode == "freq_sep":
                lo_b = _gauss(base, sigma)
                lo_p = _gauss(patch, sigma)
                patch = lo_b + (patch - lo_p) * float(detail_gain)
            elif tone_mode == "mean_std":
                ws = m.sum().clamp_min(1e-6)
                bmu = (base * m).sum(dim=(2, 3), keepdim=True) / ws
                pmu = (patch * m).sum(dim=(2, 3), keepdim=True) / ws
                bsd = ((((base - bmu) ** 2) * m).sum(dim=(2, 3), keepdim=True) / ws).sqrt().clamp_min(1e-6)
                psd = ((((patch - pmu) ** 2) * m).sum(dim=(2, 3), keepdim=True) / ws).sqrt().clamp_min(1e-6)
                patch = (patch - pmu) * (bsd / psd) + bmu
            patch = patch.clamp(0, 1)

            mw = m * w
            comp = (1.0 - mw) * base + mw * patch
            out[f, ry0:ry1, rx0:rx1, :] = comp[0].movedim(0, -1).to(out.device, dt)
            pbar.update(1)

        s = np.array(sig_list) if sig_list else np.zeros(1)
        g = np.array(mag_list) if mag_list else np.zeros(1)
        report = (f"stitch+: {B} frame(s) onto {BW}x{BH} (x{S:.2f} of {W}x{H}); tone={tone_mode} "
                  f"sigma {s.min():.1f}-{s.max():.1f}px gain {detail_gain:.2f}; canvas->box shrink "
                  f"{g.min():.2f}-{g.max():.2f}x")
        print("[H3FaceRefine+] " + report)
        return (out, report)
