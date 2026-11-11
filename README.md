# ComfyUI-H3-FaceRefine-Plus

Add-on nodes for [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine) by Carasibana, the
MiniMax H3 face refiner for video. This pack does not replace it. It works on the `H3FACEXFORM` transform that
**H3 Face Track Crop** produces, and it slots into the same workflow.

| Node | Replaces / goes after | What it fixes |
|---|---|---|
| **H3 Face Flow Lock (anti-wobble)** | after the stitch | the refined face "dancing" on the head |
| **H3 Face Stitch Plus** | replaces H3 Face Stitch | aliasing on paste, pasted-on look, low-res faces |
| **H3 Face Crop Upscale (SR)** | replaces Track Crop's crops | tiny faces reaching H3 as a blur |
| **H3 Face Tone Match** | at the end, vs the source clip | colour drift (darker, magenta skin, purple lips) from an upscaled base |
| **H3 FaceRefine Plus Strength (preset)** | into BasicScheduler's `denoise` | picking the refine strength: `standard 0.15` (recommended) or `strong 0.30` (older default) |

**Refine strength.** `standard 0.15` keeps the identity, expression and colours of the face H3 drew.
`strong 0.30` re-draws the face more: it moves toward the reference pictures, away from the face in the shot.
The higher the strength, the more the face changes. A `custom` value is also available.

## Requirements

- **ComfyUI-H3-FaceRefine** installed: H3 Face Select, H3 Face Track Crop, H3 Inject Video Latent and H3 Per-Frame Denoise come from it.
- `opencv-python` (DIS optical flow, in the main package). It is usually already present in ComfyUI.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AnniCodex/ComfyUI-H3-FaceRefine-Plus
```

Restart ComfyUI. The nodes are listed under **MiniMax H3 / Face Refine**.

## H3 Face Flow Lock: why the face wobbles

H3 Face Track Crop smooths the crop trajectory and H3 Face Stitch pastes the result back with the exact inverse
transform, so the wobble is not a tracking error. It happens inside H3. At img2img denoise ~0.3 the model re-draws
eyes, brows and mouth corners 1-2 px away from where they were, and the offset is different on every frame. Hair,
ears and the head outline keep the original pixels. The new face therefore slides on a head that does not move with
it, and better tracking cannot fix that.

Flow Lock pins the refined features back onto the base, frame by frame:

1. It computes dense optical flow, base to refined (OpenCV DIS), on lightly blurred grey frames. This follows
   structure, not the new skin texture.
2. It works only where the stitch changed the frame: a thresholded difference, dilated and feathered.
3. It smooths the flow spatially (Gaussian, mask-weighted), so it follows the misplacement and never the new detail.
   There is deliberately no temporal smoothing, because the wobble is frame-to-frame (averaging ±1 frame left half of it).
4. It warps the refined frame by that flow (bicubic) and blends it through the mask.

Wire it right after the stitch:

- `base_images`: the stitch's `base_images`.
- `refined_images`: the stitch's output.

### Measured

Test shot: MiniMax H3 music-video shot, 107 frames at 3072x1792, two people, face 206 px tall. **Wobble** is the
frame-to-frame change of the offset between the refined and the original features (phase correlation inside the
face), as a percentage of the face height.

| Version | Wobble | Face sharpness (Laplacian var.) |
|---|---|---|
| Stitch Plus, no flow lock | 0.73 % | 239 |
| **Stitch Plus + Flow Lock** | **0.19 %** | **260** |
| whole-frame latent re-render of the shot (reference, slower) | 0.38 % | n/a |

Parameters:

- `sigma_frac` (default 1/150 of the frame height) is the flow smoothing. 1/90 leaves part of the wobble. Much smaller values start to undo intentional local changes.
- `threshold` (default 4, on 0-255) sets what counts as a changed pixel. Unchanged pixels pass through.
- `base_offset` makes refined frame *k* compare with base frame `base_offset + k`.

## Compared with the official FaceRefine

Two videos in `examples/`, on the `mara` example shot (107 frames, 1536x896, face 41-62 px). Every refine uses the same
shot, prompt, reference pictures and seed. The official refine uses the values of its `H3_Face_Refine_Auto_Select`
example (crop_factor 3, denoise 0.40, stock Face Stitch, no upscale).

- `comparison_mara_official_zoom.mp4`: face zooms of the original, the official FaceRefine, FaceRefine Plus
  `standard 0.15` and FaceRefine Plus `strong 0.30`.
- `comparison_mara_strength.mp4`: the same four versions, with the whole source frame on top to show how small the
  face is in the scene.

| mara wide shot (107 f, face 41-62 px) | Wobble | Face sharpness | Colour vs original (dL / db) | Identity vs close-ups | RTX 5090 |
|---|---|---|---|---|---|
| H3 original | - | 3851 | - | 0.23 | - |
| Official FaceRefine | 0.70 % | 1613 | -0.9 / -2.2 | 0.60 | 87 s |
| FaceRefine Plus `standard 0.15` | **0.42 %** | **2979** | **-0.2 / -0.5** | 0.41 | 256 s |
| FaceRefine Plus `strong 0.30` | 0.45 % | 2948 | **-0.2 / -0.5** | **0.68** | 239 s |

Plus keeps the face steadier, sharper and in the shot's colours. At `standard` it stays close to the face H3 drew,
so it scores lower than the official refine on identity vs the close-ups. At `strong` it re-draws the face toward the
reference pictures and beats the official refine on identity as well. Doubling the strength does not add detail.
On bigger faces (about 100 px) the two presets measure almost the same.

**Many more tests and examples** (a medium-size face, method B, whole-frame side-by-sides, every setting and
measurement) are on the [FaceRefine Plus Tests](https://annicodex.github.io/FaceRefine-Plus-Tests/) page
([repository](https://github.com/AnniCodex/FaceRefine-Plus-Tests)).

## H3 Face Tone Match

Refined shots can come out darker and more magenta than the source, with purple lips. There are two causes:

1. **Video decoding (the main one).** ComfyUI's own video loader (PyAV, also used by H3 Face Select) and VHS Load Video
   (OpenCV) decode BT.709 video with the BT.601 matrix. VHS Video Combine then saves real BT.709. Even a load + save with
   no processing at all shifts the colours: measured on a 1536x896 shot, lips dL -2.6, da +1.0, db -3.2; skin dL -1.4,
   db -1.9. The same round trip with **VHS Load Video FFmpeg**, which reads the colour tags, stays within 0.25.
2. **Upscaler and H3 drift.** These are smaller, and Stitch Plus passes them on, because it takes its low frequencies
   from the base it pastes on.

Tone Match brings the result back to the source at low frequency only, in Lab:

- L follows the source with a large blur (`sigma_luma_frac`, 1/100 of the height): tone only, all detail kept.
- a, b follow it with a small blur (`sigma_chroma_frac`, 1/300): small enough for the lips. Video chroma carries no fine detail anyway.

The correction is computed at the source size and upscaled when the processed frames are larger. Measured on the same
shot, the largest Lab residual went from 2.0 to 0.6, and lips db from -1.8 to +0.1.

**Lips.** The lips are smaller than the luma blur, so after the pass above refined lips stayed about 2.3 L darker than the
source. With `lip_fix` on (the default), each face is detected (an Ultralytics face model, `face_detector`, default
`face_yolov8m.pt` in `models/ultralytics/bbox` as Impact Pack installs it). The lip pixels, meaning the reddest part of the
source's mouth area with a soft mask, get the mean L/a/b difference between source and output added as one flat offset.
The offset is smoothed over time per face, so the lips take the source's tone and every refined lip detail stays.
Measured on lip pixels only, a 106-frame shot:

| | dL | da | db | mouth sharpness |
|---|---|---|---|---|
| refined, before Tone Match | -3.98 | -1.59 | -2.62 | 1433 |
| Tone Match without lip fix | -2.35 | -1.22 | +0.19 | 1408 |
| **Tone Match with lip fix** | **-0.98** | **-0.02** | **-0.12** | 1434 |

If ultralytics or the model is missing, the lip pass is skipped and the report says so.

The table above was measured outside ComfyUI, with OpenCV decoding both sides. Here is the same shot run through
ComfyUI, measured in true colours (ffmpeg decode that reads the BT.709 tags). The 106-frame source is 1536x896:

| | lips dL | lips da | lips db | skin dL | skin db |
|---|---|---|---|---|---|
| H3 refine output (Stitch Plus) | -4.10 | -1.43 | -4.29 | -1.96 | -1.63 |
| Tone Match, `source_images` from H3 Face Select / Load Video | -3.25 | +0.34 | -3.27 | -1.69 | -1.59 |
| **Tone Match, `source_images` from Load Video FFmpeg** (example workflow) | **-0.76** | **-0.74** | **-0.12** | **-0.35** | **-0.10** |


Wire it last:

- `source_images`: the original shot, loaded with **VHS Load Video FFmpeg (path)**. Don't use the Face Select frames,
  Load Video, or ComfyUI's Load Video: they carry the same decode shift as the processed frames, so Tone Match sees no
  difference and fixes nothing.
- `processed_images`: the final refined frames, e.g. the Flow Lock output.

## H3 Face Stitch Plus

A drop-in for H3 Face Stitch, with the same `paste_region`, `mask_dilation` and `feather` options.

- **Antialiased placement.** The 768 px crop is first shrunk with an antialiased resize to the box size, then placed
  with a bicubic sub-pixel sample. The stock stitch decimates it with one bilinear `grid_sample`, which shimmers.
- **`tone_mode = freq_sep`.** Light, tone and colour come from the original frame, features and skin texture from the
  refined crop: `out = blur(base) + detail_gain * (refined - blur(refined))`. The cutoff (`lowpass_face_frac`) scales
  with the face size, so local shadows and rim light follow, and the face does not read as pasted on. `mean_std`
  (the stock global match) and `none` are kept.
- **Any base resolution.** Feed `base_images` the same clip upscaled (e.g. 2x) and the boxes are scaled to it, so the
  refined face keeps the extra pixels.
- **Optional `masks` input**, from H3 Face Mask SAM.

## H3 Face Crop Upscale (SR)

H3 Face Track Crop magnifies a small face to the 768 px canvas with one bilinear sample, so a 40 px face reaches H3 as
a smooth blob. This node rebuilds the same crops through an ESRGAN-family upscale model, so H3 starts from a sharp
face. Its temporal attention then removes the per-frame SR flicker. Settings:

- `min_magnification`: below this magnification, SR is skipped.
- `second_pass_above`: above this magnification, SR runs twice, for tiny faces only.
- `strength`: mixes the plain crop with the SR crop.

For multi-shot music videos with medium-size faces we got the best results with SR **off**, by upscaling the shot 2x
before the refine and pasting onto that 2x base with Stitch Plus.

## Examples

### Try it on the example assets

`assets/` holds four invented characters. Each folder (`mara`, `teo`, `ines`, `bruno`) is one complete run:

| File | What it is |
|---|---|
| `sample_shot.mp4` | the wide shot to refine: 1536x896, 24 fps, 17k+5 frames, with the H3 audio. The face is 45-80 px tall. |
| `sample_shot_prompt.txt` | the MiniMax H3 prompt for that shot (`<Picture 0..2>` = the three pictures below) |
| `face_reference.png` | `<Picture 0>`: the face reference |
| `character_sheet.png` | `<Picture 1>`: the character sheet |
| `keyref_source.png` | a crop of one frame of the shot: the input of the keyref edit |
| `keyref_identity_closeup.png` | a crop of the same character in a close-up of the same clip: the identity for the keyref edit |
| `keyref_qwen.png` | `<Picture 2>`: the keyref, `keyref_source.png` redrawn by Qwen Image 2.1 Edit |
| `keyref_qwen_prompt.txt` | the Qwen edit prompt used for it |
| `clip_prompt.txt` | the prompt of the whole 8 s clip (close-up + wide shot) the sample shot was cut from |
| `method_b/` | the new face reference made by method B (see below). `mara/method_b/` also has the videos. |

The workflows are set up for `mara`.

1. Copy the contents of `assets/` to `ComfyUI/input/h3plus_example/`.
2. Optional: open `face_refine_plus_keyref_qwen_workflow.json` and Queue. It rebuilds the keyref from the two crops.
3. Open `face_refine_plus_one_shot_workflow.json`. Set the full path of `mara/sample_shot.mp4` in the string node at
   the top left, then Queue. The pictures and the prompt are already set.

The characters, the images and the videos in `assets/` were made from scratch for this repo: portraits with Krea 2,
character sheets and keyrefs with Qwen Image 2.1 Edit, clips (voices included) with MiniMax H3. They do not show
real people.

### Keyref (Qwen Image 2.1 Edit, UI workflow)

`examples/face_refine_plus_keyref_qwen_workflow.json` makes the keyref:

- it takes one frame of the shot, cropped around the head;
- it takes a crop of a close-up of the same person;
- it redraws the frame sharp, with the same pose, expression and light, taking identity and detail from the close-up.

Notes in the workflow explain each step. API format: `examples/face_refine_plus_keyref_qwen_api.json`.

### Recommended: one shot, full pipeline (UI workflow)

Drag `examples/face_refine_plus_one_shot_workflow.json` onto the ComfyUI canvas. It is grouped in six steps, and each
step has a note that explains it and how to set it up:

1. Load the shot, then RTX Video Super Resolution x2.
2. Models.
3. Reference pictures (face reference, character sheet, keyref).
4. H3 Face Track Crop + H3 Reference-to-Video, Inject Video Latent, Native Audio Lock, Per-Frame Denoise.
5. Sampler: 8 turbo steps; denoise from **H3 FaceRefine Plus Strength** (`standard 0.15`; `strong 0.30` changes the face more).
6. H3 Face Stitch Plus → **H3 Face Flow Lock** → **H3 Face Tone Match** (with the lip fix) → scale back to the
   source size → Video Combine.

You only need to set:

- the shot path, and its length: H3 works on 17k+5 frames (22, 39, 56, 73, 90, 107, 124, 141...). If the shot is
  longer than one of those, trim it down with `frame_load_cap` on Face Select and on Load Video FFmpeg. A shot that
  is padded by more than a few frames gets a grid of noise on the faces of its last 17-frame group;
- the three reference pictures;
- the prompt.

The same graph in API format is `examples/face_refine_plus_one_shot_api.json`.

### Method B: keep the same face in the normal and the refined shots

*Method devised by the author of this pack.*

FaceRefine Plus redraws a small face with detail, guided by the face reference and the keyref. The refined face is
sharper, but it can look like a slightly different person from the same character in the clip's close-ups, which
H3 rendered straight from the reference. Method B makes the refined face the character's reference from the start:

1. Make the clip from the original face reference: `h3_clip_from_face_reference_workflow.json`.
2. Run FaceRefine Plus on its wide shot (keyref workflow + one-shot workflow).
3. Turn the refined face into a new neutral reference portrait: `method_b_new_face_reference_workflow.json`
   (crop of one refined frame, Qwen Image 2.1 Edit). This is `face_reference_B.png`.
4. Rebuild the character sheet from it, then re-render the clip **from scratch**: same prompt, same seed, new
   reference and new sheet.
5. Cut the new wide shot, make its keyref and run FaceRefine Plus on it again.

Do it at the start of a new video, before the other clips: it changes the face a little, so don't switch mid-project.

Measured on the four example characters (ArcFace cosine, higher = same person; *close-ups* = the close-up shot of the
same clip, *refined* = the wide shot after FaceRefine Plus):

| Character | A: wide vs close-ups | A: refined vs close-ups | B: wide vs close-ups | B: refined vs close-ups |
|---|---|---|---|---|
| mara | 0.23 | 0.41 | 0.37 | **0.50** |
| teo | 0.45 | 0.57 | 0.57 | **0.76** |
| ines | 0.30 | 0.61 | 0.36 | **0.62** |
| bruno | 0.49 | 0.62 | 0.57 | **0.74** |
| mean | 0.37 | 0.55 | 0.47 | **0.65** |

On a music video the same method took the refined face from 0.56 to 0.75. All numbers are in
`assets/method_b_results.json`. `assets/mara/method_b/` has every step: `refined_shot_A.mp4` (input of step 3),
`face_reference_B.png`, `character_sheet_B.png`, `sample_shot_B.mp4`, its keyref and `refined_shot_B.mp4`.

The two workflows have notes on every step. API format: `h3_clip_from_face_reference_api.json` and
`method_b_new_face_reference_api.json`. Besides the MiniMax H3 nodes, the clip workflow needs
[Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) and, for the
final sharpen (optional, it can be bypassed), [comfyui-vrgamedevgirl](https://github.com/vrgamegirl19/comfyui-vrgamedevgirl).

### Minimal (API format)

`examples/face_refine_plus_flowlock_api.json` is an API-format workflow:

H3 Face Select → H3 Face Track Crop → H3 Reference-to-Video + Inject Video Latent + Native Audio Lock → Per-Frame
Denoise → sampler (8 turbo steps, denoise 0.30 = the `strong` preset) → H3 Face Stitch Plus (freq_sep) → **H3 Face Flow Lock** → Video
Combine.

Before you run it, set the following to your own files and models:

- the video path;
- the prompt;
- the reference images;
- the model names.

## Credits and license

- [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine), © 2026 Carasibana, MIT License.
  This pack depends on it, and three of its helper functions are copied into `nodes_plus.py`, each marked
  `from ComfyUI-H3-FaceRefine`. See [NOTICE](NOTICE) for the full attribution and its license text.
- This pack is released under the MIT License; see [LICENSE](LICENSE).
