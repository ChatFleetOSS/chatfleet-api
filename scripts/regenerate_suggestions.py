"""
Regenerate RAG suggestions for all RAGs or a subset.

Examples:
  python -m scripts.regenerate_suggestions --all
  python -m scripts.regenerate_suggestions --rag-id <slug>
"""

import argparse
import asyncio
import time

from app.services.rags import get_collection, regenerate_suggestions_for_rag


async def _run(args: argparse.Namespace) -> int:
    col = get_collection()
    query = {}
    if args.rag_id:
        query = {"slug": args.rag_id}
    cursor = col.find(query)
    processed = ok = failed = fallback = 0
    async for rag in cursor:
        if args.limit and processed >= args.limit:
            break
        processed += 1
        slug = rag.get("slug")
        try:
            res = await regenerate_suggestions_for_rag(
                slug, force=args.force, dry_run=args.dry_run
            )
            if res.fallback_used:
                fallback += 1
            ok += 1
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"[ERROR] {slug}: {exc}")
        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000.0)
    print(f"Processed={processed} ok={ok} fallback={fallback} failed={failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate RAG suggestions")
    parser.add_argument("--all", action="store_true", help="Process all RAGs")
    parser.add_argument("--rag-id", help="Single rag slug to process")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rags (with --all)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without persisting")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between rags")
    parser.add_argument("--force", action="store_true", help="Regenerate even if version unchanged")
    args = parser.parse_args()
    if not args.all and not args.rag_id:
        parser.error("Must specify --all or --rag-id")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
