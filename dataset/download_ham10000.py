"""
Downloads HAM10000 (ISIC Archive collection 212) via the ISIC Archive REST
API v2, which returns paginated JSON (id + image URL + diagnosis metadata)
directly -- no separate metadata endpoint or HTML scraping needed.

Commands: python3 dataset/download_ham10000.py --out ../data/ham10000
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
import requests
from tqdm import tqdm

SEARCH_URL = "https://api.isic-archive.com/api/v2/images/search/?collections=212&limit=100"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download HAM10000 from the ISIC Archive REST API v2.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--user-agent", default="ham10000-downloader/2.0")
    return ap.parse_args()


def session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    return s


def fetch_records(s: requests.Session, max_images: int, timeout: int) -> list[dict]:
    """Page through the ISIC v2 search API, collecting id/url/diagnosis per image."""
    url = SEARCH_URL
    records: list[dict] = []
    pbar = tqdm(total=(max_images if max_images > 0 else None), desc="Fetching metadata", unit="img")

    while url:
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        for item in data.get("results", []):
            clinical = item.get("metadata", {}).get("clinical", {})
            full_url = item.get("files", {}).get("full", {}).get("url")
            if not full_url:
                continue
            records.append({
                "isic_id": item["isic_id"],
                "url": full_url,
                "diagnosis_1": clinical.get("diagnosis_1", ""),
                "diagnosis_2": clinical.get("diagnosis_2", ""),
                "diagnosis_3": clinical.get("diagnosis_3", ""),
            })
            pbar.update(1)
            if max_images > 0 and len(records) >= max_images:
                pbar.close()
                return records

        url = data.get("next")

    pbar.close()
    return records


def download_file(s: requests.Session, url: str, dst: Path, timeout: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with s.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def iter_jobs(records: list[dict], images_dir: Path) -> Iterable[tuple[str, str, Path]]:
    for rec in records:
        dst = images_dir / f"{rec['isic_id']}.jpg"
        yield rec["isic_id"], rec["url"], dst


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    s = session(args.user_agent)

    # Step 1: page through the search API -- id, image URL and diagnosis all
    # come back in the same JSON response, so this replaces both the old
    # HTML-crawl step and the separate metadata-download step.
    try:
        records = fetch_records(s, max_images=args.max_images, timeout=args.timeout)
    except Exception as e:
        print(f"[ERROR] Fetching metadata failed: {e}", file=sys.stderr)
        return 1

    if not records:
        print("[ERROR] No records returned from the ISIC API", file=sys.stderr)
        return 1

    print(f"[INFO] Total records fetched: {len(records)}")

    # Step 2: write metadata CSV (columns dataloader.py's _extract_ids_labels expects)
    meta_path = out_dir / "isic_ham10000_metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["isic_id", "diagnosis_1", "diagnosis_2", "diagnosis_3"])
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in ["isic_id", "diagnosis_1", "diagnosis_2", "diagnosis_3"]})
    print(f"[INFO] Saved metadata: {meta_path}")

    # Step 3: download images in parallel
    jobs = list(iter_jobs(records, images_dir))
    if args.resume:
        to_download = [j for j in jobs if not (j[2].exists() and j[2].stat().st_size > 0)]
    else:
        to_download = jobs

    print(f"[INFO] Number of images that need to be downloaded: {len(to_download)} (resume={args.resume})")

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        future_map = {
            ex.submit(download_file, s, url, dst, args.timeout): (isic_id, url, dst)
            for (isic_id, url, dst) in to_download
        }

        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="Downloading", unit="img"):
            isic_id, url, dst = future_map[fut]
            try:
                fut.result()
            except Exception:
                failures.append(isic_id)
                try:
                    if dst.exists():
                        dst.unlink()
                except Exception:
                    pass

    if failures:
        (out_dir / "failed_ids.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"[WARN] Error downloading {len(failures)} images. List saved at failed_ids.txt")
    else:
        print("[DONE] Download HAM10000 successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
