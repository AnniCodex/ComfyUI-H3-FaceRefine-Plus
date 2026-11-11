"""Quality add-ons for ComfyUI-H3-FaceRefine (Carasibana): super-resolved crops, a
detail-preserving stitch and a flow lock that stops the refined face wobbling on the head and a tone match back to the source.
Works on the H3FACEXFORM transform of H3 Face Track Crop - ComfyUI-H3-FaceRefine must be installed."""

from .nodes_plus import H3FaceCropUpscale, H3FaceStitchPlus
from .nodes_flowlock import H3FaceFlowLock
from .nodes_tonematch import H3FaceToneMatch
from .nodes_strength import H3FaceRefineStrength

NODE_CLASS_MAPPINGS = {
    "H3FaceCropUpscale": H3FaceCropUpscale,
    "H3FaceStitchPlus": H3FaceStitchPlus,
    "H3FaceFlowLock": H3FaceFlowLock,
    "H3FaceToneMatch": H3FaceToneMatch,
    "H3FaceRefineStrength": H3FaceRefineStrength,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FaceCropUpscale": "H3 Face Crop Upscale (SR)",
    "H3FaceStitchPlus": "H3 Face Stitch Plus (antialias + freq-sep + hi-res base)",
    "H3FaceFlowLock": "H3 Face Flow Lock (anti-wobble)",
    "H3FaceToneMatch": "H3 Face Tone Match (colour back to source)",
    "H3FaceRefineStrength": "H3 FaceRefine Plus Strength (preset)",
}
