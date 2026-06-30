#!/usr/bin/env python3
"""Tiny deterministic audio backend for Slurm/CLI smoke tests.

It accepts the same placeholder arguments used by the real audio templates and
writes a valid mono WAV. This lets the postprocess pipeline test command
rendering, line-vocal mixing, muxing, and manifests without requiring F5-TTS,
CosyVoice, MusicGen, or Stable Audio to be installed on the test node.
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["voice", "music", "song"], default="voice")
    parser.add_argument("--text", default="")
    parser.add_argument("--gen_text", default="")
    parser.add_argument("--music_prompt", default="")
    parser.add_argument("--voice_style", default="")
    parser.add_argument("--ref_audio", default="")
    parser.add_argument("--ref_text", default="")
    parser.add_argument("--lyrics_file", default="")
    parser.add_argument("--prompt_file", default="")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="")
    return parser.parse_args()


def _frequency(args: argparse.Namespace) -> float:
    base = {"voice": 440.0, "music": 196.0, "song": 261.63}[args.kind]
    text = args.text or args.gen_text or args.music_prompt or args.mode
    return base + (sum(ord(ch) for ch in text) + args.seed) % 80


def main() -> int:
    args = _parse_args()
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 48_000
    duration = max(args.duration, 0.1)
    samples = int(sample_rate * duration)
    frequency = _frequency(args)
    amplitude = 0.12 if args.kind == "voice" else 0.055

    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for index in range(samples):
            envelope = min(1.0, index / (sample_rate * 0.04), (samples - index) / (sample_rate * 0.08))
            value = int(32767 * amplitude * max(envelope, 0.0) * math.sin(2 * math.pi * frequency * index / sample_rate))
            wav.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
