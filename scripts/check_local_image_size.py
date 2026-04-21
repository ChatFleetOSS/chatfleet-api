from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys


def emit_github_annotation(level: str, message: str) -> None:
    print(f"::{level}::{message}")


class CountingWriter:
    def __init__(self) -> None:
        self.count = 0

    def write(self, data: bytes) -> int:
        self.count += len(data)
        return len(data)

    def flush(self) -> None:
        return None


def compute_compressed_export_size(image_ref: str) -> tuple[int | None, str | None]:
    proc = subprocess.Popen(
        ["docker", "save", image_ref],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.stdout is None or proc.stderr is None:
        proc.kill()
        return None, "failed to open docker save pipes"

    counter = CountingWriter()
    try:
        with gzip.GzipFile(fileobj=counter, mode="wb", compresslevel=1) as gz_file:
            for chunk in iter(lambda: proc.stdout.read(1024 * 1024), b""):
                gz_file.write(chunk)
    finally:
        proc.stdout.close()

    stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
    returncode = proc.wait()
    if returncode != 0:
        return None, stderr or f"docker save failed for {image_ref}"

    return counter.count, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail when a local Docker image export is too large.")
    parser.add_argument("--image", required=True, help="Local Docker image reference")
    parser.add_argument("--max-mb", type=float, required=True, help="Maximum compressed docker-save size in MB")
    args = parser.parse_args()

    proc = subprocess.run(
        ["docker", "image", "inspect", args.image],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or f"docker image inspect failed for {args.image}"
        emit_github_annotation("error", message)
        print(message, file=sys.stderr)
        return proc.returncode or 1

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        data = None
    if not data:
        message = f"No image metadata returned for {args.image}"
        emit_github_annotation("error", message)
        print(message, file=sys.stderr)
        return 1

    size_bytes, export_error = compute_compressed_export_size(args.image)
    if export_error is not None or size_bytes is None:
        message = export_error or f"unable to compute compressed export size for {args.image}"
        emit_github_annotation("error", message)
        print(message, file=sys.stderr)
        return 1

    size_mb = size_bytes / (1024 * 1024)
    message = f"[image-size] {args.image} compressed export -> {size_mb:.1f} MB"
    emit_github_annotation("notice", message)
    print(message)

    if size_mb > args.max_mb:
        error_message = (
            f"[image-size][error] {args.image} is {size_mb:.1f} MB, above limit {args.max_mb:.1f} MB"
        )
        emit_github_annotation("error", error_message)
        print(error_message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
