"""H3 FaceRefine Plus Strength: one place to pick how hard the refine re-draws the face.

The value goes into the denoise input of the sampler's BasicScheduler. It is the base strength: H3 Per-Frame Denoise
still multiplies it per frame (1.0/1.0 in the Plus graphs, so the value is used as it is)."""

PRESETS = {
    "standard 0.15 (recommended: keeps the face H3 drew)": 0.15,
    "strong 0.30 (older default: re-draws the face more, toward the references)": 0.30,
}
CUSTOM = "custom (use the value below)"


class H3FaceRefineStrength:
    """Refine strength preset for the sampler's denoise. Higher = more re-drawing of the face."""

    DESCRIPTION = ("Refine strength (sampler denoise). 'standard 0.15' keeps the identity, expression and colours of "
                   "the face H3 drew. 'strong 0.30' is the older default: it re-draws the face more, so the "
                   "face moves toward the reference pictures (closer to the character, further from the shot). WARNING: the higher the "
                   "strength, the more the face is changed. Connect 'denoise' to the denoise input of BasicScheduler.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(PRESETS) + [CUSTOM], {
                    "default": next(iter(PRESETS)),
                    "tooltip": "standard 0.15 = faithful to the shot (recommended). strong 0.30 = the face is re-drawn "
                               "more and moves toward the references (identity, expression, make-up). Raising the strength always "
                               "changes the face more."}),
                "custom_denoise": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Used only with preset 'custom'. Above 0.30 the face is re-drawn even "
                               "more and can stop looking like the person in the shot."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("denoise", "note")
    FUNCTION = "run"
    CATEGORY = "H3/FaceRefine Plus"

    def run(self, preset, custom_denoise):
        d = float(custom_denoise) if preset == CUSTOM else PRESETS[preset]
        note = f"refine strength (denoise) {d:.2f}"
        if d > 0.15:
            note += ": stronger than the standard 0.15, the face will change more"
        return (d, note)
