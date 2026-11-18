"""H3 FaceRefine Plus Strength: one place to pick how hard the refine re-draws the face.

The value goes into the denoise input of the sampler's BasicScheduler. It is the base strength: H3 Per-Frame Denoise
still multiplies it per frame (1.0/1.0 in the Plus graphs, so the value is used as it is).
Since v0.2.0 the default is 0.30 (0.15 changed too little). The new H3 FaceRefine Plus Mode node picks the strength
AND the pipeline (V1 / V2, VOSR2) in one dropdown; this node stays for the V1-only workflows."""

PRESETS = {
    "0.30 (default: re-draws the face toward the references)": 0.30,
    "0.15 (light: keeps more of the face H3 drew)": 0.15,
}
CUSTOM = "custom (use the value below)"
# labels saved by v0.1.x workflows
LEGACY = {
    "standard 0.15 (recommended: keeps the face H3 drew)": 0.15,
    "strong 0.30 (older default: re-draws the face more, toward the references)": 0.30,
}


class H3FaceRefineStrength:
    """Refine strength preset for the sampler's denoise. Higher = more re-drawing of the face."""

    DESCRIPTION = ("Refine strength (sampler denoise). '0.30' (default) re-draws the face toward the reference "
                   "pictures: identity closer to the character (mara example: 0.69 vs 0.41 at 0.15). '0.15' keeps more "
                   "of the face H3 drew. WARNING: the higher the strength, the more the face is changed. Connect "
                   "'denoise' to the denoise input of BasicScheduler.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(PRESETS) + [CUSTOM], {
                    "default": next(iter(PRESETS)),
                    "tooltip": "0.30 = default, the face moves toward the references (identity, expression, make-up). "
                               "0.15 = lighter, closer to the shot. Raising the strength always changes the face more."}),
                "custom_denoise": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Used only with preset 'custom'. Above 0.30 the face is re-drawn even "
                               "more and can stop looking like the person in the shot."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("denoise", "note")
    FUNCTION = "run"
    CATEGORY = "H3/FaceRefine Plus"

    @classmethod
    def VALIDATE_INPUTS(cls, preset):
        # also accept the labels of v0.1.x workflows
        return True if preset in PRESETS or preset in LEGACY or preset == CUSTOM else f"unknown preset: {preset}"

    def run(self, preset, custom_denoise):
        d = float(custom_denoise) if preset == CUSTOM else PRESETS.get(preset, LEGACY.get(preset, 0.30))
        note = f"refine strength (denoise) {d:.2f}"
        if d > 0.30:
            note += ": stronger than the default 0.30, the face will change more"
        return (d, note)
