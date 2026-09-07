"""Command line entry point: whiteboard-animate."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .regions import SnippetRegionPlan
from .render import Scene, render_video


def _load_plan(path: str | None) -> SnippetRegionPlan | None:
    if not path:
        return None
    return SnippetRegionPlan.model_validate_json(Path(path).read_text())


def _detect_plan(image: Path, narration: str, model: str | None) -> SnippetRegionPlan:
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "--detect-regions needs the gemini extra: pip install 'whiteboard-animator[gemini]'"
        ) from exc
    from .detect_regions import DEFAULT_MODEL, detect_region_plan

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("--detect-regions needs GOOGLE_API_KEY in the environment")
    client = genai.Client(api_key=api_key)
    return asyncio.run(detect_region_plan(
        google_client=client,
        image_path=image,
        transcript_text=narration,
        model=model or DEFAULT_MODEL,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whiteboard-animate",
        description="Animate whiteboard-style images as hand-drawn reveal videos.",
    )
    parser.add_argument("images", nargs="+", help="scene images in order")
    parser.add_argument("-o", "--output", required=True, help="output MP4")
    parser.add_argument("--audio", nargs="*", default=[],
                        help="one narration file per image, in the same order")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="seconds per scene when there is no audio (default 8)")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--regions", nargs="*", default=[],
                        help="one region plan JSON per image, in the same order")
    parser.add_argument("--detect-regions", action="store_true",
                        help="ask Gemini for a region plan per image (needs GOOGLE_API_KEY)")
    parser.add_argument("--narration", nargs="*", default=[],
                        help="narration text per image, used by --detect-regions")
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--save-regions", action="store_true",
                        help="write detected plans next to the output as <output>.regions.<n>.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    n = len(args.images)
    if args.audio and len(args.audio) != n:
        raise SystemExit(f"expected {n} audio files, got {len(args.audio)}")
    if args.regions and len(args.regions) != n:
        raise SystemExit(f"expected {n} region plans, got {len(args.regions)}")
    if args.regions and args.detect_regions:
        raise SystemExit("use --regions or --detect-regions, not both")
    if args.narration and len(args.narration) != n:
        raise SystemExit(f"expected {n} narration strings, got {len(args.narration)}")

    scenes = []
    for index, image in enumerate(args.images):
        plan = _load_plan(args.regions[index]) if args.regions else None
        if args.detect_regions:
            narration = args.narration[index] if args.narration else ""
            plan = _detect_plan(Path(image), narration, args.gemini_model)
            if args.save_regions:
                plan_path = f"{args.output}.regions.{index}.json"
                Path(plan_path).write_text(plan.model_dump_json(indent=2))
                print(f"wrote {plan_path}", file=sys.stderr)
        scenes.append(Scene(
            image=image,
            audio=args.audio[index] if args.audio else None,
            duration=None if args.audio else args.duration,
            region_plan=plan,
        ))

    durations = render_video(scenes, args.output, quality=args.quality)
    print(f"wrote {args.output} ({sum(durations):.1f}s, {len(durations)} scene(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
