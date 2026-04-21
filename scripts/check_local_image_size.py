from __future__ import annotations

import argparse
import json
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail when a local Docker image is too large.")
    parser.add_argument("--image", required=True, help="Local Docker image reference")
    parser.add_argument("--max-mb", type=float, required=True, help="Maximum uncompressed image size in MB")
    args = parser.parse_args()

    proc = subprocess.run(
        ["docker", "image", "inspect", args.image],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr.strip() or f"docker image inspect failed for {args.image}", file=sys.stderr)
        return proc.returncode or 1

    data = json.loads(proc.stdout)
    if not data:
        print(f"No image metadata returned for {args.image}", file=sys.stderr)
        return 1

    size_bytes = int(data[0]["Size"])
    size_mb = size_bytes / (1024 * 1024)
    print(f"[image-size] {args.image} -> {size_mb:.1f} MB")

    if size_mb > args.max_mb:
        print(
            f"[image-size][error] {args.image} is {size_mb:.1f} MB, above limit {args.max_mb:.1f} MB",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

