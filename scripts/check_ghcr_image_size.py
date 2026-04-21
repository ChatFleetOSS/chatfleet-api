from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request


OCI_INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}


def parse_image_ref(image_ref: str) -> tuple[str, str, str]:
    if ":" not in image_ref.rsplit("/", 1)[-1]:
        raise ValueError(f"Image ref must include a tag: {image_ref}")
    registry_and_repo, tag = image_ref.rsplit(":", 1)
    registry, repo = registry_and_repo.split("/", 1)
    return registry, repo, tag


def fetch_token(repo: str) -> str:
    token_url = f"https://ghcr.io/token?scope=repository:{repo}:pull"
    with urllib.request.urlopen(token_url) as resp:
        payload = json.load(resp)
    return payload["token"]


def fetch_manifest(repo: str, ref: str, token: str) -> tuple[dict, str]:
    req = urllib.request.Request(
        f"https://ghcr.io/v2/{repo}/manifests/{urllib.parse.quote(ref, safe=':')}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.index.v1+json, "
                "application/vnd.docker.distribution.manifest.list.v2+json, "
                "application/vnd.oci.image.manifest.v1+json, "
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
        },
    )
    with urllib.request.urlopen(req) as resp:
        media_type = resp.headers.get("Content-Type", "").split(";", 1)[0]
        payload = json.load(resp)
    return payload, media_type


def compressed_size_mb(manifest: dict) -> float:
    return sum(layer.get("size", 0) for layer in manifest.get("layers", [])) / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check compressed GHCR image size per platform.")
    parser.add_argument("--image-ref", required=True, help="Full image ref, e.g. ghcr.io/org/repo:tag")
    parser.add_argument(
        "--max-compressed-mb",
        type=float,
        required=True,
        help="Maximum compressed size per platform manifest in MB",
    )
    args = parser.parse_args()

    registry, repo, tag = parse_image_ref(args.image_ref)
    if registry != "ghcr.io":
        print(f"Only ghcr.io is supported, got {registry}", file=sys.stderr)
        return 1

    token = fetch_token(repo)
    root_manifest, media_type = fetch_manifest(repo, tag, token)

    platform_sizes: list[tuple[str, float]] = []
    if media_type in OCI_INDEX_TYPES:
        for child in root_manifest.get("manifests", []):
            platform = child.get("platform", {})
            os_name = platform.get("os")
            arch = platform.get("architecture")
            if not os_name or not arch or os_name == "unknown" or arch == "unknown":
                continue
            manifest, _ = fetch_manifest(repo, child["digest"], token)
            platform_sizes.append((f"{os_name}/{arch}", compressed_size_mb(manifest)))
    else:
        platform_sizes.append(("single", compressed_size_mb(root_manifest)))

    if not platform_sizes:
        print(f"No platform manifests found for {args.image_ref}", file=sys.stderr)
        return 1

    failed = False
    for platform_name, size_mb in platform_sizes:
        print(f"[ghcr-image-size] {args.image_ref} {platform_name} -> {size_mb:.1f} MB compressed")
        if size_mb > args.max_compressed_mb:
            failed = True

    if failed:
        print(
            f"[ghcr-image-size][error] {args.image_ref} exceeds {args.max_compressed_mb:.1f} MB compressed "
            "on at least one platform",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
