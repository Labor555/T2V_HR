#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="", help="Fallback prompt/caption for every clip.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.video_root)
    pattern = "**/*" if args.recursive else "*"
    videos = sorted(path for path in root.glob(pattern) if path.suffix.lower() in VIDEO_EXTS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for path in videos:
            handle.write(json.dumps({"video": str(path), "prompt": args.prompt}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(videos)} videos to {output}")


if __name__ == "__main__":
    main()

