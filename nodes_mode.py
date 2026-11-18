"""H3 FaceRefine Plus Mode + H3 FaceRefine Plus Switch: one dropdown picks the whole face refine recipe.

The example workflow "face_refine_plus_modes" holds both pipelines:
  V1 - the FaceRefine Plus graph: whole frame x2 (RTX VSR), face crop re-drawn with prompt + 3 references,
       Stitch Plus, Flow Lock (anti-wobble), Tone Match. Best identity and the least wobble in our tests.
  V2 - SAM3 mask on the face (or face + hair) -> H3 Face Mask To Latent: only the masked part of the crop is re-drawn,
       empty prompt + the character sheet, optional VOSR2 on the refined crops. Faster, but more wobble.
The Mode node gives the denoise, the SAM3 text and two switches; the Switch nodes are lazy, so only the
branch you picked runs (with V1 nothing of V2 is executed, VOSR2 included, and the other way round).

Measured on the mara example (107 frames, 1536x896, face 41-62 px, RTX 5090, only the model loaders cached):
  mode                       identity  wobble  sharpness  time
  V1 (0.30)                  0.69      0.43%   2957       91 s
  V2 (0.30 face)             0.56      0.57%   2096       62 s
  V2 (0.30 face) + VOSR2     0.57      0.56%   2767       ~116 s
  V2 (0.30 face + hair)      0.56      0.65%   2019       63 s
  V2 (0.50 face + hair)+VOSR2 0.63     0.80%   3299       123 s
VOSR2 is offered only for V2: on V1 it made the face softer (2957 -> 2522) and took 31 s more."""

MODES = {
    # label: (branch, denoise, sam_text, vosr2)
    "V1 (0.30) - default": ("v1", 0.30, "face", False),
    "V1 (0.15)": ("v1", 0.15, "face", False),
    "V2 (0.30 face)": ("v2", 0.30, "face", False),
    "V2 (0.50 face)": ("v2", 0.50, "face", False),
    "V2 (0.30 face + hair)": ("v2", 0.30, "face and hair", False),
    "V2 (0.50 face + hair)": ("v2", 0.50, "face and hair", False),
    "V2 (0.30 face) + VOSR2": ("v2", 0.30, "face", True),
    "V2 (0.50 face) + VOSR2": ("v2", 0.50, "face", True),
    "V2 (0.30 face + hair) + VOSR2": ("v2", 0.30, "face and hair", True),
    "V2 (0.50 face + hair) + VOSR2": ("v2", 0.50, "face and hair", True),
}
DEFAULT = "V1 (0.30) - default"


class H3FaceRefineMode:
    DESCRIPTION = ("Pick the face refine recipe. V1 = the FaceRefine Plus graph (prompt + 3 refs, Flow Lock, Tone Match): "
                   "best identity, least wobble. V2 = SAM3 face (or face + hair) mask, only that part is re-drawn, "
                   "empty prompt + sheet: faster, more wobble; '+ VOSR2' sharpens the refined crops (~+55 s, needs ComfyUI-VOSR2). "
                   "Higher denoise = the face moves more toward the references. Wire: denoise -> both BasicSchedulers, "
                   "sam_text -> the SAM3 CLIPTextEncode, use_v2 -> the final H3 FaceRefine Plus Switch, use_vosr2 -> the VOSR2 switch.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(MODES), {"default": DEFAULT,
                         "tooltip": "V1 (0.30) is the default and the best in our tests. V1 (0.15) changes the face less. "
                                    "V2 re-draws only the SAM3 mask: 'face' or 'face + hair'; 0.50 = stronger. "
                                    "'+ VOSR2' only exists for V2 (on V1 it made the face softer)."}),
            },
            "optional": {
                "custom_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "0 = use the mode's denoise. Any other value replaces it."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("denoise", "sam_text", "use_v2", "use_vosr2", "note")
    FUNCTION = "run"
    CATEGORY = "H3/FaceRefine Plus"

    def run(self, mode, custom_denoise=0.0):
        branch, d, txt, vosr = MODES.get(mode, MODES[DEFAULT])
        if custom_denoise and custom_denoise > 0:
            d = float(custom_denoise)
        note = f"{mode}: {branch.upper()}, denoise {d:.2f}" + (f", SAM3 '{txt}'" if branch == "v2" else "") + (", VOSR2" if vosr else "")
        print("[H3FaceRefine+] mode " + note)
        return (d, txt, branch == "v2", bool(vosr), note)


class H3FaceRefineSwitch:
    """Lazy IMAGE switch: only the selected input is computed (the other branch does not run)."""

    DESCRIPTION = ("Passes on_true when 'select' is true, else on_false. Lazy: the branch that is not selected is not "
                   "executed. Use it with H3 FaceRefine Plus Mode (use_v2 / use_vosr2).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("BOOLEAN", {"default": False, "forceInput": True}),
            },
            "optional": {
                "on_false": ("IMAGE", {"lazy": True}),
                "on_true": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "H3/FaceRefine Plus"

    def check_lazy_status(self, select, on_false=None, on_true=None):
        if select:
            return ["on_true"] if on_true is None else []
        return ["on_false"] if on_false is None else []

    def run(self, select, on_false=None, on_true=None):
        out = on_true if select else on_false
        if out is None:
            raise ValueError("H3 FaceRefine Plus Switch: the selected input is not connected.")
        return (out,)
