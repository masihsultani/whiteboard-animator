# whiteboard-animator

Turn a whiteboard-style image into a hand-drawn reveal video.

Give it a picture of a finished whiteboard (marker sketch, diagram, hand lettering, generated or scanned) and it produces an MP4 where the picture draws itself the way a person would: text is written word by word, outlines are traced, solid shapes are outlined then filled with brush sweeps, and branched line art is drawn one stroke at a time. Add a narration file and the drawing paces itself to the audio.

This is the render engine behind the Whiteboard format at [Kinoslide](https://kinoslide.com), extracted so it can run on its own.

## How it works

1. **Detect components.** Ink pixels are split into connected components. A [CRAFT](https://github.com/clovaai/CRAFT-pytorch) text detector (bundled ONNX model, runs on CPU) marks which components are text.
2. **Order them.** Components are grouped spatially and sorted the way a hand would draw: containers before contents, shapes before their labels, reading order for text.
3. **Schedule.** Each component gets a time slot proportional to the square root of its area. With a region plan, slots follow the narration instead.
4. **Trace.** Every pixel gets a reveal time. Strokes follow their skeleton from an endpoint. Closed outlines use one contour front. Fills get an outline pass then an angled sweep or bristled brush strokes. Multi-branch line art is decomposed into sequential pen paths.
5. **Encode.** Frames stream to ffmpeg. Only the pixels currently fading are touched per frame, so a 1280px scene encodes in about a second.

## Install

Needs Python 3.10+ and `ffmpeg` and `ffprobe` on your PATH.

```bash
pip install whiteboard-animator
# optional: Gemini-based region detection
pip install 'whiteboard-animator[gemini]'
```

From a checkout:

```bash
pip install -e '.[dev]'
pytest
```

## Use

One image, fixed length:

```bash
whiteboard-animate scene.png --duration 8 -o scene.mp4
```

One image with narration. The drawing finishes inside the audio and the finished frame holds until the audio ends:

```bash
whiteboard-animate scene.png --audio scene.wav -o scene.mp4
```

Several scenes concatenated in order:

```bash
whiteboard-animate a.png b.png c.png --audio a.wav b.wav c.wav -o lecture.mp4
```

Quality presets are `low` (20 fps, 500k), `medium` (24 fps, 1500k, default) and `high` (24 fps, 3000k). Images larger than 1280px on the long side are downscaled.

### Region plans

By default the whole image draws over the first 70% of the scene in a heuristic order. A region plan tells the animator what the image contains, in what order to draw it, and which narration phrase goes with each part. Regions then draw in windows sized by how long their phrase takes to say, so the drawing follows the voice.

```bash
whiteboard-animate scene.png --audio scene.wav --regions scene.regions.json -o scene.mp4
```

A plan is JSON with normalized 0 to 1000 boxes (top-left origin). See `examples/photosynthesis.regions.json`:

```json
{
  "idea": "Photosynthesis in one picture",
  "regions": [
    {
      "label": "title",
      "role": "title",
      "object": "the word Photosynthesis",
      "reveal_order": 1,
      "box": {"ymin": 60, "xmin": 40, "ymax": 180, "xmax": 700},
      "expected_visual": "Title text",
      "annotation": "Photosynthesis is how plants make food.",
      "reveal": "stroke"
    }
  ]
}
```

With the `gemini` extra and `GOOGLE_API_KEY` set, the plan can be detected from the image and narration text:

```bash
whiteboard-animate scene.png --audio scene.wav \
  --detect-regions --narration "Photosynthesis is how plants make food. ..." \
  --save-regions -o scene.mp4
```

`--save-regions` writes the detected plan next to the output so you can edit it and re-render with `--regions`.

## Python API

```python
from whiteboard_animator import Scene, SnippetRegionPlan, render_video

plan = SnippetRegionPlan.model_validate_json(open("scene.regions.json").read())
render_video(
    [Scene("a.png", audio="a.wav", region_plan=plan), Scene("b.png", audio="b.wav")],
    "lecture.mp4",
    quality="high",
)
```

Lower level, `WhiteboardAnimator.render_to_file(img_array, draw_duration, total_duration, output_path, fps=24, bitrate="1500k", element_plan=None)` takes an RGB numpy array and writes a silent MP4. The constructor exposes every tuning knob: fade length, fill detection thresholds, brush angle and width, the S-curve that decides when a fill uses brush strokes instead of a sweep, and the line-art decomposition thresholds.

## Input images

The engine assumes ink on a white background. Anything lighter than 240 gray is treated as background, and near-white pixels are snapped to white. Images with a colored or textured background will not animate well. Clean marker-style renders with 3 to 6 colors work best.

## Text detection model

`whiteboard_animator/models/craft.onnx` (83 MB) is an ONNX export of `craft_mlt_25k` from [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) (MIT). Set `CRAFT_MODEL_PATH` to point at a different file. If the model cannot be loaded the animator still runs, it just treats text as ordinary strokes.

## License

MIT.
