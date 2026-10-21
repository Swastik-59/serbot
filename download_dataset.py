"""
HaGRID annotation downloader for GestureGrasp.

The model trains on hand landmarks, and HaGRID v2's annotations_with_landmarks archive already holds
21 MediaPipe landmarks for every labelled hand. So no images are needed: this script reads the remote
zip's directory with HTTP range requests (remotezip) and fetches only the train/val/test JSON files
for the classes training uses, as listed in configs/intents.yaml (gestures, aliases, hard negatives).

    python download_dataset.py --check     # list files, sizes and the on-disk total; download nothing
    python download_dataset.py             # fetch whatever is missing

Refuses to grow data/raw past --hard-cap-gb on disk. Writes reports/download_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import yaml
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

try:
    from remotezip import RangeNotSupported, RemoteIOError, RemoteZip
except ImportError:
    print("Missing dependency. Run: pip install remotezip requests tqdm", file=sys.stderr)
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("download_dataset")

ANNOTATIONS_URL = ("https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
                   "annotations_with_landmarks/annotations.zip")
SPLITS = ("train", "val", "test")
USER_AGENT = "GestureGrasp-Downloader/3.0"
REQUEST_TIMEOUT = (20, 180)  # (connect, read)
# SberCloud OBS intermittently resets the TLS handshake (WinError 10054); retry with a fresh session.
MAX_ATTEMPTS = 6
BACKOFF_BASE_SECONDS = 2.0


def gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(total=4, connect=4, read=4, backoff_factor=0.6, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["HEAD", "GET"]), respect_retry_after_header=True)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _is_transient(exc: BaseException) -> bool:
    """True for dropped connections and server-side blips, not for missing ranges or 4xx."""
    if isinstance(exc, RangeNotSupported):
        return False
    if isinstance(exc, requests.exceptions.HTTPError):
        code = exc.response.status_code if exc.response is not None else 0
        return code == 429 or code >= 500
    return isinstance(exc, (requests.exceptions.RequestException, RemoteIOError, TimeoutError, ConnectionError, OSError))


class ResilientRemoteZip:
    """RemoteZip that opens a fresh session after a connection reset and retries member reads."""

    def __init__(self, url: str):
        self.url = url
        self._rz: Optional[RemoteZip] = None
        self._session: Optional[requests.Session] = None

    def _close_handle(self) -> None:
        for obj in (self._rz, self._session):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self._rz, self._session = None, None

    def _retry(self, action: str, fn):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return fn()
            except Exception as exc:
                self._close_handle()
                if not _is_transient(exc) or attempt >= MAX_ATTEMPTS:
                    raise
                delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning("%s failed (attempt %d/%d): %s. Retrying in %.0fs...",
                               action, attempt, MAX_ATTEMPTS, exc, delay)
                time.sleep(delay)

    def _ensure(self) -> RemoteZip:
        if self._rz is None:
            self._session = make_session()
            self._rz = RemoteZip(self.url, session=self._session, timeout=REQUEST_TIMEOUT)
        return self._rz

    def infolist(self):
        return self._retry("Opening archive", lambda: self._ensure().infolist())

    def read(self, filename: str) -> bytes:
        return self._retry(f"Reading {filename}", lambda: self._ensure().read(filename))

    def close(self) -> None:
        self._close_handle()

    def __enter__(self) -> "ResilientRemoteZip":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def training_classes(intents_path: Path) -> list[str]:
    """Every HaGRID class training reads: the gestures, their aliases and the hard negatives."""
    cfg = yaml.safe_load(Path(intents_path).read_text(encoding="utf-8"))
    extra = cfg.get("training_data", {}) or {}
    names = list(cfg["gestures"]) + list((extra.get("aliases") or {})) + list(extra.get("hard_negatives") or [])
    return list(dict.fromkeys(names))


def list_remote(cache_dir: Path, refresh: bool) -> list[dict]:
    """[{filename, compress_size, file_size}] for every JSON in the remote archive (cached locally)."""
    cache = cache_dir / "annotations_listing.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    logger.info("Reading the remote archive's directory (a small range read, not a download)...")
    with ResilientRemoteZip(ANNOTATIONS_URL) as rz:
        entries = [{"filename": zi.filename, "compress_size": zi.compress_size, "file_size": zi.file_size}
                   for zi in rz.infolist() if zi.filename.lower().endswith(".json") and "checkpoint" not in zi.filename]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(entries), encoding="utf-8")
    return entries


def plan(classes: list[str], entries: list[dict]) -> tuple[list[tuple[dict, str]], list[str]]:
    """(entry, split) for each class x split, plus any that the archive lacks."""
    by_key = {}
    for e in entries:
        parts = Path(e["filename"]).parts
        split = next((s for s in SPLITS if s in parts), None)
        if split:
            by_key.setdefault((Path(e["filename"]).stem, split), e)
    chosen = [(by_key[(c, s)], s) for c in classes for s in SPLITS if (c, s) in by_key]
    missing = [f"{s}/{c}.json" for c in classes for s in SPLITS if (c, s) not in by_key]
    return chosen, missing


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HaGRID annotation JSON (hand landmarks) for training.")
    parser.add_argument("--check", action="store_true", help="Show the plan and sizes; download nothing.")
    parser.add_argument("--classes", nargs="+", help="Override the class list (default: from configs/intents.yaml).")
    parser.add_argument("--intents", default="configs/intents.yaml")
    parser.add_argument("--hard-cap-gb", type=float, default=10.0, help="Never let data/raw grow past this on disk.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--refresh-listing", action="store_true", help="Re-read the remote directory.")
    parser.add_argument("--redownload", action="store_true", help="Fetch files even if already present.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_root = raw_dir / "annotations"
    classes = args.classes or training_classes(Path(args.intents))
    try:
        entries = list_remote(raw_dir / "_listing_cache", args.refresh_listing)
    except RangeNotSupported:
        raise SystemExit("The HaGRID host stopped supporting HTTP range requests; partial download is impossible.")
    chosen, missing = plan(classes, entries)
    if missing:
        logger.warning("Not in the remote archive: %s", ", ".join(missing))

    todo = [(e, s) for e, s in chosen
            if args.redownload or not (out_root / s / Path(e["filename"]).name).exists()]
    on_disk = dir_size(raw_dir)
    growth = sum(e["file_size"] for e, _ in todo)
    hard_cap = int(args.hard_cap_gb * 1024 ** 3)
    print(f"{len(classes)} classes x {len(SPLITS)} splits: {len(chosen)} files, {len(todo)} to fetch "
          f"({gb(sum(e['compress_size'] for e, _ in todo))} transfer).")
    print(f"data/raw: {gb(on_disk)} now -> {gb(on_disk + growth)} after (hard cap {gb(hard_cap)}).")
    if on_disk + growth > hard_cap:
        raise SystemExit("Refused: data/raw would exceed --hard-cap-gb on disk.")
    if args.check or not todo:
        return

    fetched = 0
    with ResilientRemoteZip(ANNOTATIONS_URL) as rz:
        for entry, split in tqdm(todo, desc="annotations", unit="file"):
            target = out_root / split / Path(entry["filename"]).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(rz.read(entry["filename"]))
            fetched += 1

    manifest = {"timestamp": datetime.now(timezone.utc).isoformat(), "classes": classes,
                "files_fetched": fetched, "files_planned": len(chosen), "missing": missing,
                "data_raw_bytes": dir_size(raw_dir)}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Fetched %d files; data/raw is now %s.", fetched, gb(manifest["data_raw_bytes"]))


if __name__ == "__main__":
    main()
