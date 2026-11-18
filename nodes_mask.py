"""H3 Face Mask To Latent: turns per-frame face masks (e.g. SAM3 "face" on the H3 Face Track Crop crops) into a
noise mask on the H3 video latent grid, so the sampler re-draws ONLY the face (use it with DifferentialDiffusion).

Why: measured on a 34 px face (LLQ C46 s1), refining the whole crop at denoise 0.30 smeared the eyes and raised the
extra flicker from 6.6 to 8.4; with the face-only latent mask the same refine can go to denoise 0.5 without touching hair,
background or the crop borders.

Steps (all on the GPU/CPU tensor, no extra models):
 1. clean   - threshold, then an opening (shrink, grow back + edge_grow): removes specks, gives a margin around the face;
              frames where the mask drops out for up to `fill_gap` frames get the nearest good mask.
 2. grow    - extra spatial growth in pixels (the latent cell is 16 px: a face edge inside a cell must re-draw the cell).
 3. to latent - max-pool to the latent grid (T', H', W') read from the latent itself, then grow `grow_temporal` latent
              frames both ways. Max, not average: a latent cell is re-drawn if ANY of its pixels is face.
Output: the same latent with its noise_mask set (replaces SetLatentNoiseMask), plus the mask for preview.
Written for ComfyUI-H3-FaceRefine-Plus, MIT licence."""
import torch
import torch.nn.functional as F
import comfy.model_management


def _dilate(x, r):
    """x: (T, H, W) in 0..1, square dilation of radius r px. Separable (rows, then columns): the same result as one
    (2r+1)^2 max-pool at a fraction of the cost."""
    if r <= 0:
        return x
    k = 2 * r + 1
    y = F.max_pool2d(x[:, None], (1, k), stride=1, padding=(0, r))
    return F.max_pool2d(y, (k, 1), stride=1, padding=(r, 0))[:, 0]


def _erode(x, r):
    return 1.0 - _dilate(1.0 - x, r)


def _fill_gaps(x, gap):
    """frames with an empty mask take the nearest non-empty mask within `gap` frames"""
    if gap <= 0:
        return x
    has = x.flatten(1).amax(1) > 0
    if bool(has.all()) or not bool(has.any()):
        return x
    idx = torch.nonzero(has)[:, 0]
    out = x.clone()
    for t in torch.nonzero(~has)[:, 0].tolist():
        d = (idx - t).abs()
        j = int(torch.argmin(d))
        if int(d[j]) <= gap:
            out[t] = x[int(idx[j])]
    return out


class H3FaceMaskToLatent:
    DESCRIPTION = ("Face masks (SAM3 'face' on the crops) -> noise mask on the H3 video latent: only the face is re-drawn. "
                   "Connect the VAE-encoded crops as 'latent' and use the output latent in place of SetLatentNoiseMask; "
                   "add DifferentialDiffusion on the model. With the mask, denoise 0.15-0.5 stays on the face only.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "VAE Encode of the face crops (the video latent, before the audio concat)"}),
                "masks": ("MASK", {"tooltip": "one mask per crop frame, same size as the crops (SAM3_TrackToMask)"}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shrink": ("INT", {"default": 12, "min": 0, "max": 128, "tooltip": "opening radius in px: removes specks smaller than this"}),
                "edge_grow": ("INT", {"default": 16, "min": 0, "max": 256, "tooltip": "margin added around the face after the opening (px)"}),
                "fill_gap": ("INT", {"default": 4, "min": 0, "max": 64, "tooltip": "frames where the mask drops out for up to this many frames take the nearest mask"}),
                "grow_spatial": ("INT", {"default": 36, "min": 0, "max": 256, "tooltip": "extra growth in px before going to the latent grid"}),
                "grow_temporal": ("INT", {"default": 1, "min": 0, "max": 8, "tooltip": "latent frames added before and after the masked ones"}),
            }
        }

    RETURN_TYPES = ("LATENT", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("latent", "latent_mask", "clean_masks", "report")
    FUNCTION = "run"
    CATEGORY = "H3/FaceRefine Plus"

    def run(self, latent, masks, threshold, shrink, edge_grow, fill_gap, grow_spatial, grow_temporal):
        s = latent["samples"]
        if s.ndim != 5:
            raise ValueError(f"H3FaceMaskToLatent: expected a video latent (B,C,T,H,W), got shape {tuple(s.shape)}")
        T2, H2, W2 = s.shape[2], s.shape[3], s.shape[4]
        dev = comfy.model_management.get_torch_device()
        m = masks.to(dev).float()
        if m.ndim == 2:
            m = m[None]
        m = (m > threshold).float()
        m = _dilate(_erode(m, shrink), shrink + edge_grow)
        m = _fill_gaps(m, fill_gap)
        clean = m
        m = _dilate(m, grow_spatial)
        lm = F.adaptive_max_pool3d(m[None, None], (T2, H2, W2))[0, 0]
        if grow_temporal > 0:
            k = 2 * grow_temporal + 1
            lm = F.max_pool1d(lm.permute(1, 2, 0).reshape(-1, 1, T2), k, stride=1, padding=grow_temporal).reshape(H2, W2, T2).permute(2, 0, 1)
        out = dict(latent)
        lm, clean = lm.cpu(), clean.cpu()
        out["noise_mask"] = lm.reshape((-1, 1, H2, W2))
        cover = float(lm.mean())
        empty = int((clean.flatten(1).amax(1) == 0).sum())
        report = (f"face mask -> latent {T2}x{H2}x{W2}: {cover * 100:.1f}% of the latent re-drawn, "
                  f"{empty}/{clean.shape[0]} crop frames without a face mask")
        if empty > clean.shape[0] // 4:
            report += " | WARNING: the face mask is missing in many frames (check the SAM3 text and the crops)"
        if cover < 0.02:
            report += " | WARNING: almost nothing will be re-drawn (mask too small or missing)"
        print("[H3FaceRefine+] " + report)
        return (out, lm, clean, report)
