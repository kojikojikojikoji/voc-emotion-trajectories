"""Download the DailyDialog dataset archive.

Usage:
    python scripts/download_data.py

The zip (about 4.5 MB, 13,118 dialogues with per-utterance emotion labels)
lands at ``$VOC_DATA_DIR/raw/ijcnlp_dailydialog.zip`` (default
``./data/raw/``; override ``VOC_DATA_DIR`` in a ``.env`` file). The download
is idempotent: an existing complete file is kept.

License notice: DailyDialog (Li et al., IJCNLP 2017) is released under
CC BY-NC-SA 4.0. That license permits redistribution with attribution under
the same terms, which is why a small sample is committed under
``data/sample/``; the full archive stays out of the repository only to keep
it lean.

Source notes (2026-07): the original host ``yanran.li`` is a parked domain
and its download URL returns an HTML placeholder, and the Hugging Face
mirror ``li2017dailydialog/daily_dialog`` is a loading script that points at
the same dead URL. The Internet Archive capture of the original zip
(2022-05-16) is therefore the primary source; the historical URL is kept as
a fallback in case the domain comes back.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voc_arc.data import raw_zip_path  # noqa: E402

URLS = [
    "https://web.archive.org/web/20220516002703id_/http://yanran.li/files/ijcnlp_dailydialog.zip",
    "http://yanran.li/files/ijcnlp_dailydialog.zip",
]

EXPECTED_SIZE = 4_475_921  # bytes, for a sanity check after download
CHUNK = 1 << 20


def _fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": "voc-emotion-trajectories"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as fh:
        while chunk := response.read(CHUNK):
            fh.write(chunk)
    tmp.replace(dest)


def _looks_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < EXPECTED_SIZE * 0.9:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        return "ijcnlp_dailydialog/dialogues_emotion.txt" in names
    except zipfile.BadZipFile:
        return False


def main() -> int:
    dest = raw_zip_path()
    if _looks_complete(dest):
        print(f"already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("License: DailyDialog, CC BY-NC-SA 4.0 (Li et al., IJCNLP 2017).")
    for attempt, url in enumerate([u for u in URLS for _ in range(2)]):
        try:
            print(f"downloading {url}")
            _fetch(url, dest)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  failed: {exc}")
            time.sleep(2**attempt)
            continue
        if not _looks_complete(dest):
            print(f"  not a valid DailyDialog zip ({dest.stat().st_size} bytes), retrying")
            continue
        print(f"done: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return 0
    print(
        "Could not download the dataset. Check your network connection, or fetch "
        "ijcnlp_dailydialog.zip from an archive of http://yanran.li/dailydialog and "
        "place it at data/raw/ijcnlp_dailydialog.zip manually. The notebooks fall "
        "back to the committed data/sample/ if the raw file is absent."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
