"""Detect a region plan from a finished whiteboard image with Gemini.

Optional. Requires the `gemini` extra and GOOGLE_API_KEY.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

from .regions import Box, Region, RegionRole, RevealStyle, SnippetRegionPlan


DEFAULT_MODEL = "gemini-3.1-flash-lite"
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
GEMINI_RETRY_DELAY_SEC = float(os.getenv("GEMINI_RETRY_DELAY_SEC", "3"))

logger = logging.getLogger(__name__)


class DetectedRegion(BaseModel):
    label: str = Field(description="Short slug describing the region's content, e.g. 'title', 'co2_arrow'.")
    role: RegionRole
    object: str = Field(description="Short description of the exact thing in this region.")
    reveal_order: int = Field(description="1-indexed draw order. UNIQUE across the array.")
    box: Box = Field(description="Named normalized bounding box: ymin/xmin/ymax/xmax, each 0-1000.")
    annotation: str = Field(
        default="",
        description="Transcript phrase or short paraphrase spoken while this region should draw.",
    )
    reveal: RevealStyle = Field(description="Reveal style hint: 'stroke', 'fill', or 'fade'.")


class DetectionResponse(BaseModel):
    regions: List[DetectedRegion]


def _is_retryable_gemini_error(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code == 429 or (isinstance(status_code, int) and status_code >= 500)


async def _generate_content_with_retry(
    google_client: genai.Client,
    *,
    operation: str,
    **kwargs,
):
    last_exc: Exception | None = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            return await google_client.aio.models.generate_content(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= GEMINI_MAX_RETRIES or not _is_retryable_gemini_error(exc):
                raise
            delay = GEMINI_RETRY_DELAY_SEC * attempt
            logger.warning(
                "Gemini %s failed on attempt %s/%s with %s; retrying in %.1fs",
                operation,
                attempt,
                GEMINI_MAX_RETRIES,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise last_exc if last_exc is not None else RuntimeError(f"Gemini {operation} failed without an exception")


def _build_prompt(transcript_text: str, idea: str) -> str:
    return f"""You are analyzing a finished whiteboard-style educational illustration.

Task: detect reveal regions in this finished whiteboard image. A region is one
meaningful drawing act before the presenter moves elsewhere. The animator
reveals regions in `reveal_order` and uses local drawing heuristics inside each
region.

Snippet idea: "{idea}"
Narration that plays over this scene: "{transcript_text}"

For each region produce:
  - label:           short slug describing the region's content.
  - role:            one of title, main_concept, supporting_detail, annotation, label, decoration.
  - object:          short description of exactly what this region contains.
  - reveal_order:    1-indexed integer. UNIQUE across the array.
  - box:             object with named fields ymin, xmin, ymax, xmax, normalized 0-1000.
                     ymin < ymax, xmin < xmax. Top-left origin.
  - annotation:      the transcript phrase or short paraphrase that matches this region.
  - reveal:          one of "stroke", "fill", "fade".
                     * "stroke" — line art, text, arrows, outlines.
                     * "fill"   — solid filled shapes, color blocks.
                     * "fade"   — backgrounds or faint decorations.

GRANULARITY RULES:

- Text regions must include the FULL contiguous text string exactly as drawn.
  If the visible text is "Mass (M)", "Charge (Q)", or "Spin (J)", the box must
  enclose the word, spaces, parentheses, and symbol together. Do not box only
  the first word.
- Each distinct text line is usually one region: title, subtitle, bullet line,
  equation line, axis label, legend item, callout label, caption, or theorem text.
- Do not split a single text line into separate word/symbol regions unless the
  symbols are physically far apart and clearly drawn as separate labels.
- Split large composite visuals into logical drawing passes. Do not make a
  whole chart, cell, flowchart, machine, balance-sheet diagram, or object
  drawing one giant region when it contains teachable parts.
- For large objects, make the first region the anchor structure, then separate
  regions for meaningful internal parts, callouts, arrows, labels, or details
  the narration visits later.
- Keep small simple icons as one region, but split any visual that would feel
  unnatural if a teacher drew it all at once.
- Connector arrows/brackets/lines should be separate regions when they express
  the relationship between already-existing regions.
- Avoid tiny fragments and giant multi-beat regions. A region should be one
  meaningful drawing act.

DECOMPOSITION EXAMPLES:

- Cell diagram: large cell outline first, nucleus/mitochondria/ER/Golgi as
  separate regions, then labels or arrows.
- Graph: axes first, each curve or bar group next, intersection markers after,
  labels/callouts last.
- Flowchart/system: boxes or objects first in narration order, connector arrows
  after both endpoints exist, labels last.
- Cause/effect scene: cause object first, effect object next, arrow or force
  line after both are visible.

REVEAL ORDER RULES:

- DO NOT default to pure geometry. Read the narration and choose the teaching
  sequence: what the presenter says first should generally draw first.
- Within the same narration beat, use a natural hand path: anchor/title first,
  main object next, labels and arrows after the thing they describe.
- When text labels annotate a diagram, draw the diagram or containing object
  before its label unless the label is the thing being taught.
- If uncertain, choose the order that makes the image understandable at every
  intermediate reveal step.
"""


async def detect_region_plan(
    *,
    google_client: genai.Client,
    image_path: Path,
    transcript_text: str,
    idea: str = "",
    model: str = DEFAULT_MODEL,
) -> SnippetRegionPlan:
    from PIL import Image as PILImage
    img = PILImage.open(image_path)
    w, h = img.size
    image_bytes = image_path.read_bytes()

    prompt = _build_prompt(transcript_text, idea or "(untitled)")

    response = await _generate_content_with_retry(
        google_client,
        operation="region detection",
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DetectionResponse,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    parsed: DetectionResponse | None = response.parsed if response else None
    if parsed is None or not parsed.regions:
        raise RuntimeError(f"Region detection returned no regions. raw: {(response.text or '')[:500]}")

    regions = [
        Region(
            label=d.label,
            role=d.role,
            object=d.object,
            reveal_order=d.reveal_order,
            box=d.box,
            expected_visual=d.object,
            annotation=d.annotation,
            reveal=d.reveal,
        )
        for d in parsed.regions
    ]

    return SnippetRegionPlan(
        idea=idea or "(untitled)",
        global_style_notes="",
        regions=regions,
        image_size=[w, h],
    )
