"""Video encoding backend.

Converts the animator's per-pixel reveal-time map into an MP4 by streaming
raw frames to ffmpeg.

Performance:
- Frame rendering uses sorted-pixel bucketing: only the narrow band of
  actively-fading pixels is blended each frame; fully-revealed pixels are
  committed once to a persistent buffer.  ~50-100x fewer pixel ops per frame.
- Pre-allocated frame buffer eliminates per-frame array allocation.
- The hold phase is generated with ffmpeg tpad instead of piping duplicate
  frames through stdin.
"""

import logging
import subprocess
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def ffmpeg_cmd(w, h, fps, bitrate, preset, output_path):
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", preset, "-b:v", bitrate, "-an",
        output_path,
    ]


def ffmpeg_cmd_tpad(w, h, fps, bitrate, preset, output_path, hold_dur):
    """Build ffmpeg command that clones the last frame for hold_dur seconds."""
    stop_frames = int(hold_dur * fps)
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-vf", f"tpad=stop_mode=clone:stop={stop_frames}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", preset, "-b:v", bitrate, "-an",
        output_path,
    ]


def encode_blank_video(out_w, out_h, total_duration, output_path, fps, bitrate, preset):
    white = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
    white_bytes = white.tobytes()

    cmd = ffmpeg_cmd(out_w, out_h, fps, bitrate, preset, output_path)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    total_frames = int(total_duration * fps)
    for _ in range(total_frames):
        proc.stdin.write(white_bytes)
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode()}")


def encode_timing_video(
        img_array, timing, draw_end, total_duration, fade_dur,
        output_path, fps, bitrate, preset,
):
    """Stream a timed pixel-reveal of img_array to an MP4.

    timing holds each pixel's reveal time in seconds (inf = background).
    Only the narrow band of actively-fading pixels is blended per frame;
    fully-revealed pixels are committed once to a persistent buffer.
    """
    t2 = time.monotonic()
    h, w = img_array.shape[:2]
    out_h, out_w = h + (h % 2), w + (w % 2)

    # ── Gather content pixels ──
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    cy, cx = np.where(gray < 240)
    ct_timing = timing[cy, cx]
    ct_colors_f = img_array[cy, cx].astype(np.float32)
    ct_colors_u = img_array[cy, cx]  # uint8, for final commit

    # ── Sort pixels by reveal time ──
    order = np.argsort(ct_timing, kind="quicksort")
    s_timing = ct_timing[order]
    s_cy = cy[order]
    s_cx = cx[order]
    s_colors_f = ct_colors_f[order]
    s_colors_u = ct_colors_u[order]
    del ct_timing, ct_colors_f, ct_colors_u, cy, cx, order

    # ── Persistent frame buffer — fully-revealed pixels accumulate here ──
    frame_buf = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    committed = 0  # everything below this index is baked into frame_buf

    t3 = time.monotonic()
    logger.info("frame prep + sort: %.2fs", t3 - t2)

    # ── Encode ──
    # Render only the draw portion + a small buffer, then use ffmpeg
    # tpad to clone the last frame for the hold phase.  This avoids
    # piping gigabytes of identical hold frames through stdin.
    hold_dur = max(total_duration - draw_end - fade_dur, 0)
    draw_frames = int((draw_end + fade_dur) * fps) + 1

    use_tpad = hold_dur > 0.5  # worth it if > ~12 duplicate frames

    if use_tpad:
        cmd = ffmpeg_cmd_tpad(
            out_w, out_h, fps, bitrate, preset, output_path, hold_dur
        )
    else:
        draw_frames = int(total_duration * fps)
        cmd = ffmpeg_cmd(out_w, out_h, fps, bitrate, preset, output_path)

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    frame_view = memoryview(frame_buf).cast("B")

    for i in range(draw_frames):
        t_now = i / fps

        # 1. Commit newly fully-revealed pixels (their fade is done)
        full_cutoff = t_now - fade_dur
        new_committed = int(np.searchsorted(s_timing, full_cutoff, side="right"))
        if new_committed > committed:
            sl = slice(committed, new_committed)
            frame_buf[s_cy[sl], s_cx[sl]] = s_colors_u[sl]
            committed = new_committed

        # 2. Blend the narrow band of actively-fading pixels
        active_end = int(np.searchsorted(s_timing, t_now, side="right"))

        if active_end > committed:
            sl = slice(committed, active_end)
            a = np.clip((t_now - s_timing[sl]) / fade_dur, 0, 1)
            np.sqrt(a, out=a)
            a3 = a[:, np.newaxis]
            frame_buf[s_cy[sl], s_cx[sl]] = (
                    255.0 + (s_colors_f[sl] - 255.0) * a3
            ).astype(np.uint8)

        # Reuse the same buffer view instead of allocating a multi-megabyte
        # bytes object for every frame.
        proc.stdin.write(frame_view)

    proc.stdin.close()
    proc.wait()
    t4 = time.monotonic()
    logger.info("frame pipe + encode: %.2fs (%d frames piped)", t4 - t3, draw_frames)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.read().decode()}")
