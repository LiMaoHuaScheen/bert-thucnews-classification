#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download THUCNews arrow files (10-class Chinese text classification dataset).

Source: seamew/THUCNewsText on HuggingFace, mirrored via hf-mirror.com.
Files are Arrow STREAM format; load with pyarrow.ipc.open_stream.
"""
import os
import subprocess
import sys

BASE = "https://hf-mirror.com/datasets/seamew/THUCNewsText/resolve/main"
FILES = ["thuc_news_text-train.arrow", "thuc_news_text-validation.arrow", "thuc_news_text-test.arrow"]
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    os.makedirs(DEST, exist_ok=True)
    for f in FILES:
        path = os.path.join(DEST, f)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[skip] {f} already exists")
            continue
        url = f"{BASE}/{f}"
        print(f"[download] {f} <- {url}")
        subprocess.run(["curl", "-s", "-L", "-o", path, url], check=True)
        print(f"[ok] {f} {os.path.getsize(path)} bytes")
    print("Done. Data in", DEST)


if __name__ == "__main__":
    main()
