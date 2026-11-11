"""H3 Face Tone Match: bring a processed clip's tone and colour back to its source, keeping the processed detail.

Main cause of the drift we measured: ComfyUI's video loader (PyAV, used by H3 Face Select) and VHS Load Video
(OpenCV) decode BT.709 video with the BT.601 matrix, while VHS Video Combine saves real BT.709, so load + save alone
turns faces darker/magenta and lips purple (lips dL -2.6 da +1.0 db -3.2 with no processing). Upscaler and H3 drift
add a little more; Stitch Plus takes its low frequencies from the base, so the refined face inherits all of it.
=> feed source_images from VHS Load Video FFmpeg (it reads the colour tags). A source decoded the same wrong way as
the processed frames shows no difference and nothing is fixed.

Per frame, in CIE Lab at the SOURCE size:
  L += blur(L_src - L_out, sigma_luma)     large sigma: tone/exposure only, every luma detail is kept
  a += blur(a_src - a_out, sigma_chroma)   small sigma: follows even the lips; chroma carries no fine detail
  b += blur(b_src - b_out, sigma_chroma)
If the processed frames are larger than the source, the correction is computed at the source size and upscaled.
Lips (lip_fix, on by default): lips are smaller than the luma blur, so refined lips stayed ~2.3 L darker than the
source. Per face (Ultralytics face detector, e.g. face_yolov8m.pt from Impact Pack's ultralytics/bbox), the lip
pixels - the reddest part of the SOURCE mouth area, soft mask - get the mean L/a/b difference source - output as a flat
offset (EMA over time per face): the lips take the source's tone, every refined lip detail stays. Lips dL -2.35 -> -1.1.
Skipped, with a note in the report, if ultralytics or the detector model is not available.
"""

from __future__ import annotations

import numpy as np
import torch

import comfy.utils
import comfy.model_management as mm


class H3FaceToneMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE", {"tooltip": "The original frames (any size). Load them with VHS Load Video FFmpeg: ComfyUI Load Video / VHS Load Video / H3 Face Select decode BT.709 as BT.601, so the colour shift would be invisible here."}),
                "processed_images": ("IMAGE", {"tooltip": "The refined / upscaled frames, same content (same or larger size)."}),
                "sigma_luma_frac": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001,
                    "tooltip": "Lightness correction blur, fraction of the SOURCE height (0.01 = 9 px at 896)."}),
                "sigma_chroma_frac": ("FLOAT", {"default": 1.0 / 300.0, "min": 0.0005, "max": 0.05, "step": 0.0005,
                    "tooltip": "Colour correction blur, fraction of the SOURCE height (1/300 = 3 px at 896)."}),
                "source_offset": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "processed frame k matches source frame source_offset + k."}),
            },
            "optional": {
                "lip_fix": ("BOOLEAN", {"default": True, "tooltip": "Bring each face's lips to the source's tone (needs ultralytics)."}),
                "face_detector": ("STRING", {"default": "face_yolov8m.pt",
                    "tooltip": "File in models/ultralytics/bbox (Impact Pack layout)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = "Tone and colour back to the source at low frequency (fixes upscale/refine colour drift)."

    @staticmethod
    def _detector(name):
        try:
            import folder_paths
            from ultralytics import YOLO
            path = None
            for key in ("ultralytics_bbox", "ultralytics"):
                try:
                    path = folder_paths.get_full_path(key, name if key == "ultralytics_bbox" else "bbox/" + name)
                except Exception:
                    path = None
                if path:
                    break
            return (YOLO(path), "") if path else (None, f"lip fix skipped: {name} not found in models/ultralytics/bbox")
        except Exception as exc:
            return None, f"lip fix skipped: {exc}"

    @staticmethod
    def _lip_masks(det, src_rgb01, lab):
        import cv2
        bgr = (src_rgb01[..., ::-1] * 255.0 + 0.5).astype(np.uint8)
        out = []
        boxes = det.predict(bgr, conf=0.4, verbose=False, device="cpu")[0].boxes.xyxy.tolist()
        for x0, y0, x1, y1 in sorted(boxes, key=lambda b: b[0] + b[2]):
            w, h = x1 - x0, y1 - y0
            if h < 40:
                out.append(None); continue
            cx, cy, rx, ry = (x0 + x1) / 2, y0 + 0.79 * h, 0.20 * w, 0.10 * h
            Y0, Y1, X0, X1 = int(cy - 2 * ry), int(cy + 2 * ry), int(cx - 2 * rx), int(cx + 2 * rx)
            if Y0 < 0 or X0 < 0 or Y1 > lab.shape[0] or X1 > lab.shape[1]:
                out.append(None); continue
            yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
            ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            av = lab[Y0:Y1, X0:X1, 1]
            m = np.clip((av - np.percentile(av[ell], 62)) / 3.0, 0, 1) * ell
            m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, h / 120))
            out.append(((Y0, Y1, X0, X1), m) if m.sum() > 10 else None)
        return out

    def run(self, source_images, processed_images, sigma_luma_frac, sigma_chroma_frac, source_offset,
            lip_fix=True, face_detector="face_yolov8m.pt"):
        import cv2
        det, note = self._detector(face_detector) if lip_fix else (None, "")
        ema, lipd = {}, []

        n = processed_images.shape[0]
        if source_images.shape[0] < source_offset + n:
            raise ValueError(f"source_images has {source_images.shape[0]} frames, need source_offset + {n}.")
        H, W = source_images.shape[1:3]
        QH, QW = processed_images.shape[1:3]
        sl, sc = max(0.5, sigma_luma_frac * H), max(0.5, sigma_chroma_frac * H)
        out = processed_images[..., :3].clone()
        dev = []
        pbar = comfy.utils.ProgressBar(n)
        for k in range(n):
            mm.throw_exception_if_processing_interrupted()
            s = source_images[source_offset + k, ..., :3].float().cpu().numpy()
            q = processed_images[k, ..., :3].float().cpu().numpy()
            qs = cv2.resize(q, (W, H), interpolation=cv2.INTER_AREA) if (QW, QH) != (W, H) else q
            d = cv2.cvtColor(s, cv2.COLOR_RGB2LAB) - cv2.cvtColor(qs, cv2.COLOR_RGB2LAB)
            corr = np.dstack([cv2.GaussianBlur(d[..., 0], (0, 0), sl)] +
                             [cv2.GaussianBlur(d[..., c], (0, 0), sc) for c in (1, 2)])
            dev.append(np.abs(corr).reshape(-1, 3).mean(0))
            if det is not None:
                ls = cv2.cvtColor(s, cv2.COLOR_RGB2LAB)
                after = cv2.cvtColor(qs, cv2.COLOR_RGB2LAB) + corr
                for f, lm in enumerate(self._lip_masks(det, s, ls)):
                    if lm is None:
                        continue
                    (Y0, Y1, X0, X1), m = lm
                    w8 = m / m.sum()
                    dl = np.array([((ls[Y0:Y1, X0:X1, c] - after[Y0:Y1, X0:X1, c]) * w8).sum() for c in range(3)], np.float32)
                    ema[f] = dl if f not in ema else 0.5 * ema[f] + 0.5 * dl
                    corr[Y0:Y1, X0:X1] += ema[f] * np.clip(m * 1.6, 0, 1)[..., None]
                    lipd.append(ema[f])
            if (QW, QH) != (W, H):
                corr = cv2.resize(corr, (QW, QH), interpolation=cv2.INTER_LINEAR)
            res = np.clip(cv2.cvtColor(cv2.cvtColor(q, cv2.COLOR_RGB2LAB) + corr, cv2.COLOR_LAB2RGB), 0, 1)
            out[k] = torch.from_numpy(res).to(out.dtype)
            pbar.update(1)
        m = np.mean(dev, 0) if dev else np.zeros(3)
        report = f"tone match: {n} frame(s), mean |correction| L {m[0]:.2f} a {m[1]:.2f} b {m[2]:.2f}"
        if lipd:
            lm = np.mean(lipd, 0)
            report += f"; lips {len(lipd)} face-frame(s), mean offset L {lm[0]:+.2f} a {lm[1]:+.2f} b {lm[2]:+.2f}"
        elif note:
            report += "; " + note
        print("[H3FaceRefine+] " + report)
        return (out, report)
