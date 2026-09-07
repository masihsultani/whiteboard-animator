"""Turn a whiteboard-style image into a hand-drawn reveal video."""

from .animator import WhiteboardAnimator
from .regions import Box, Region, SnippetRegionPlan, build_narration_weighted_plan
from .render import Scene, render_scene, render_video

__all__ = [
    "Box",
    "Region",
    "Scene",
    "SnippetRegionPlan",
    "WhiteboardAnimator",
    "build_narration_weighted_plan",
    "render_scene",
    "render_video",
]
