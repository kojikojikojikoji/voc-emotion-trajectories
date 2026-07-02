"""Download the datasets used by the notebooks.

Usage:
    python scripts/download_data.py [dailydialog|p4g|all]

Without an argument both datasets are fetched. Files land under
``$VOC_DATA_DIR/raw/`` (default ``./data/raw/``; override ``VOC_DATA_DIR``
in a ``.env`` file). Downloads are idempotent: existing complete files are
kept.

DailyDialog (notebooks 01-03): the zip (about 4.5 MB, 13,118 dialogues with
per-utterance emotion labels) lands at ``raw/ijcnlp_dailydialog.zip``.
License: CC BY-NC-SA 4.0 (Li et al., IJCNLP 2017). That license permits
redistribution with attribution under the same terms, which is why a small
sample is committed under ``data/sample/``. Source notes (2026-07): the
original host ``yanran.li`` is a parked domain and its download URL returns
an HTML placeholder, and the Hugging Face mirror
``li2017dailydialog/daily_dialog`` is a loading script that points at the
same dead URL. The Internet Archive capture of the original zip
(2022-05-16) is therefore the primary source; the historical URL is kept as
a fallback in case the domain comes back.

Persuasion for Good (notebook 04): four files (about 3.9 MB total) from the
official GitLab repository ``ucdavisnlp/persuasionforgood`` land under
``raw/persuasionforgood/``: the full 1,017-dialogue corpus
(``full_dialog.csv``), the participant info table with the actual donations
(``full_info.csv``), and the 300-dialogue annotated subset
(``300_dialog.xlsx`` plus ``300_info.xlsx``). License: Apache-2.0 (Wang et
al., ACL 2019), which permits the committed sample under ``data/sample/``.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voc_arc import p4g  # noqa: E402
from voc_arc.data import raw_zip_path  # noqa: E402

DAILYDIALOG_URLS = [
    "https://web.archive.org/web/20220516002703id_/http://yanran.li/files/ijcnlp_dailydialog.zip",
    "http://yanran.li/files/ijcnlp_dailydialog.zip",
]

P4G_BASE = "https://gitlab.com/ucdavisnlp/persuasionforgood/-/raw/master"
P4G_FILES = [
    ("data/FullData/full_dialog.csv", p4g.FULL_DIALOG_CSV_NAME),
    ("data/FullData/full_info.csv", p4g.FULL_INFO_CSV_NAME),
    ("data/AnnotatedData/300_dialog.xlsx", p4g.ANNOTATED_XLSX_NAME),
    ("data/AnnotatedData/300_info.xlsx", "300_info.xlsx"),
]

EXPECTED_ZIP_SIZE = 4_475_921  # bytes, for a sanity check after download
CHUNK = 1 << 20
ATTEMPTS = 3


def _fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": "voc-emotion-trajectories"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as fh:
        while chunk := response.read(CHUNK):
            fh.write(chunk)
    tmp.replace(dest)


def _fetch_with_retries(urls: list[str], dest: Path, valid: Callable[[Path], bool]) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt, url in enumerate([u for u in urls for _ in range(ATTEMPTS)]):
        try:
            print(f"downloading {url}")
            _fetch(url, dest)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  failed: {exc}")
            time.sleep(2 ** min(attempt, 4))
            continue
        if not valid(dest):
            print(f"  not a valid file ({dest.stat().st_size} bytes), retrying")
            continue
        print(f"done: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    return False


def _dailydialog_zip_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < EXPECTED_ZIP_SIZE * 0.9:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        return "ijcnlp_dailydialog/dialogues_emotion.txt" in names
    except zipfile.BadZipFile:
        return False


def _csv_with_header(header_token: str, min_size: int) -> Callable[[Path], bool]:
    def check(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size < min_size:
            return False
        with open(path, encoding="utf-8", errors="replace") as fh:
            return header_token in fh.readline()

    return check


def download_dailydialog() -> bool:
    dest = raw_zip_path()
    if _dailydialog_zip_complete(dest):
        print(f"already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    print("License: DailyDialog, CC BY-NC-SA 4.0 (Li et al., IJCNLP 2017).")
    if _fetch_with_retries(DAILYDIALOG_URLS, dest, _dailydialog_zip_complete):
        return True
    print(
        "Could not download DailyDialog. Check your network connection, or fetch "
        "ijcnlp_dailydialog.zip from an archive of http://yanran.li/dailydialog and "
        "place it at data/raw/ijcnlp_dailydialog.zip manually. Notebooks 01-03 fall "
        "back to the committed data/sample/ if the raw file is absent."
    )
    return False


def download_p4g() -> bool:
    print("License: Persuasion for Good, Apache-2.0 (Wang et al., ACL 2019).")
    checks: dict[str, Callable[[Path], bool]] = {
        p4g.FULL_DIALOG_CSV_NAME: _csv_with_header("Unit", 1_000_000),
        p4g.FULL_INFO_CSV_NAME: _csv_with_header("B6", 100_000),
        p4g.ANNOTATED_XLSX_NAME: p4g.looks_like_annotated_xlsx,
        "300_info.xlsx": lambda path: path.is_file() and path.stat().st_size > 10_000,
    }
    ok = True
    for remote, name in P4G_FILES:
        dest = p4g.p4g_raw_dir() / name
        valid = checks[name]
        if valid(dest):
            print(f"already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        if not _fetch_with_retries([f"{P4G_BASE}/{remote}"], dest, valid):
            ok = False
    if not ok:
        print(
            "Could not download Persuasion for Good completely. Check your network "
            "connection, or fetch the files from "
            "https://gitlab.com/ucdavisnlp/persuasionforgood (data/FullData and "
            "data/AnnotatedData) into data/raw/persuasionforgood/ manually. "
            "Notebook 04 falls back to the committed data/sample/ if the raw files "
            "are absent."
        )
    return ok


def main(argv: list[str]) -> int:
    choice = argv[1] if len(argv) > 1 else "all"
    if choice not in ("dailydialog", "p4g", "all"):
        print(__doc__)
        return 2
    ok = True
    if choice in ("dailydialog", "all"):
        ok = download_dailydialog() and ok
    if choice in ("p4g", "all"):
        ok = download_p4g() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
