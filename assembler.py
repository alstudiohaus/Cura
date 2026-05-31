"""
Cura video assembler using FFmpeg.
Takes a sequence of image paths and assembles a 9:16 MP4.
"""

import subprocess
import os
import uuid
import tempfile
from typing import List, Optional


def assemble_reel(
    image_paths: List[str],
    output_path: Optional[str] = None,
    duration_seconds: int = 30,
    music_path: Optional[str] = None,
    transition: str = "crossfade",
) -> str:
    """
    Assemble a list of images into a 9:16 reel MP4.

    Args:
        image_paths: Ordered list of image file paths
        output_path: Output MP4 path (auto-generated if None)
        duration_seconds: Total reel duration
        music_path: Optional background music file
        transition: crossfade | slide | none

    Returns:
        Path to the output MP4 file
    """
    if not image_paths:
        raise ValueError("No images provided")

    if output_path is None:
        output_path = f"/tmp/cura_{uuid.uuid4().hex[:8]}.mp4"

    clip_duration = duration_seconds / len(image_paths)
    width, height = 1080, 1920  # 9:16

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write FFmpeg input file list
        list_path = os.path.join(tmpdir, "inputs.txt")
        with open(list_path, "w") as f:
            for path in image_paths:
                f.write(f"file '{path}'\n")
                f.write(f"duration {clip_duration:.2f}\n")

        if transition == "crossfade":
            cmd = _build_crossfade_cmd(image_paths, output_path, clip_duration, width, height, music_path)
        else:
            cmd = _build_simple_cmd(list_path, output_path, width, height, music_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

    return output_path


def _build_simple_cmd(list_path, output_path, width, height, music_path):
    """Simple concat without transitions."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            "format=yuv420p"
        ),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
    ]
    if music_path:
        cmd += ["-i", music_path, "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd.append(output_path)
    return cmd


def _build_crossfade_cmd(image_paths, output_path, clip_duration, width, height, music_path):
    """
    Build FFmpeg crossfade command.
    Uses xfade filter between consecutive clips.
    """
    if len(image_paths) == 1:
        return _build_simple_cmd(image_paths[0], output_path, width, height, music_path)

    fade_duration = min(0.5, clip_duration * 0.2)
    inputs = []
    for path in image_paths:
        inputs += ["-loop", "1", "-t", str(clip_duration), "-i", path]

    # Scale filter for each input
    scale_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},format=yuv420p,setsar=1"
    )

    filter_parts = []
    # Scale all inputs
    for i in range(len(image_paths)):
        filter_parts.append(f"[{i}:v]{scale_filter}[v{i}]")

    # Chain xfade
    prev = "v0"
    for i in range(1, len(image_paths)):
        offset = clip_duration * i - fade_duration * i
        out = f"xf{i}" if i < len(image_paths) - 1 else "out"
        filter_parts.append(
            f"[{prev}][v{i}]xfade=transition=fade:duration={fade_duration}:offset={offset:.2f}[{out}]"
        )
        prev = out

    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
    ]

    if music_path:
        cmd += ["-i", music_path, "-c:a", "aac", "-b:a", "192k", "-shortest"]

    cmd.append(output_path)
    return cmd


def apply_ken_burns(image_path: str, output_path: str, duration: float = 3.0, width: int = 1080, height: int = 1920):
    """Apply Ken Burns (zoom/pan) effect to a single image."""
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-vf", (
            f"scale={width*2}:{height*2},"
            f"zoompan=z='min(zoom+0.0015,1.5)':d={int(duration*25)}:s={width}x{height},"
            "format=yuv420p"
        ),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path
