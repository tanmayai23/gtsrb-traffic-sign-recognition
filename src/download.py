"""Fetch and unpack the GTSRB archives.

The mirror is a university server that is occasionally slow, so downloads are
resumable: a partial file is kept as ``<name>.part`` and continued with a Range
request rather than restarted.
"""

from __future__ import annotations

import os
import socket
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

# Sizes recorded from the mirror. Used as a checksum substitute -- the hosts have
# re-zipped these archives before, which changes the hash but not the contents,
# so an exact MD5 would produce false alarms.
ARCHIVES = {
    "GTSRB_Final_Training_Images.zip": 276_294_756,
    "GTSRB_Final_Test_Images.zip": 88_978_620,
    "GTSRB_Final_Test_GT.zip": 99_620,
}

SIZE_TOLERANCE = 0.02
USER_AGENT = "Mozilla/5.0 (compatible; gtsrb-cli/1.0)"

# Counted from the archives themselves; the training zip also holds 43 CSV files
# and directory entries, so its 39,299 entries are NOT the image count.
EXPECTED_TRAIN_IMAGES = 39_209
EXPECTED_TEST_IMAGES = 12_630


def _size_ok(actual: int, expected: int | None) -> bool:
    if expected is None:
        return True
    return abs(actual - expected) <= expected * SIZE_TOLERANCE


def download_file(
    url: str,
    dest: Path,
    expected_size: int | None = None,
    retries: int = 3,
    timeout: int = 60,
) -> Path:
    """Download ``url`` to ``dest``, resuming a partial transfer if one exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and _size_ok(dest.stat().st_size, expected_size):
        print(f"  [cached] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # A server that ignores our Range header restarts the body at 0,
                # so the local partial has to be discarded to avoid a corrupt file.
                if have and resp.status != 206:
                    have = 0
                    part.unlink(missing_ok=True)

                remaining = resp.headers.get("Content-Length")
                total = (int(remaining) + have) if remaining else expected_size

                mode = "ab" if have else "wb"
                with open(part, mode) as fh, tqdm(
                    total=total,
                    initial=have,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"  {dest.name}",
                    ncols=80,
                ) as bar:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                        bar.update(len(chunk))

            got = part.stat().st_size
            if not _size_ok(got, expected_size):
                raise OSError(
                    f"size mismatch for {dest.name}: got {got}, expected ~{expected_size}"
                )

            os.replace(part, dest)  # atomic: a crash never leaves a half file at dest
            return dest

        except (urllib.error.URLError, OSError, socket.timeout) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download {url} after {retries} attempts: {exc}\n"
                    f"You can download it manually and place it at: {dest}"
                ) from exc
            wait = 2**attempt
            print(f"  ! attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)

    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> int:
    """Extract ``zip_path`` into ``dest_dir``, refusing entries that escape it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for name in members:
            target = (root / name).resolve()
            # Guard against zip-slip: a crafted entry like "../../evil" would
            # otherwise write outside the extraction directory.
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"unsafe path in {zip_path.name}: {name}")
        zf.extractall(dest_dir)

    return len(members)


def _count_ppm(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.ppm"))


def ensure_dataset(data_dir: Path, force: bool = False) -> Path:
    """Download and extract all three archives. Returns the extraction root."""
    raw_dir = Path(data_dir) / "raw"
    extract_dir = Path(data_dir) / "extracted"
    sentinel = extract_dir / ".extracted_ok"

    if force:
        sentinel.unlink(missing_ok=True)

    if sentinel.exists():
        print(f"[prepare] dataset already extracted at {extract_dir}")
        return extract_dir

    print("[prepare] downloading GTSRB archives (~365 MB total)")
    for name, size in ARCHIVES.items():
        download_file(BASE_URL + name, raw_dir / name, expected_size=size)

    print("[prepare] extracting")
    for name in ARCHIVES:
        count = extract_zip(raw_dir / name, extract_dir)
        print(f"  {name}: {count} entries")

    train_root = extract_dir / "GTSRB" / "Final_Training" / "Images"
    test_root = extract_dir / "GTSRB" / "Final_Test" / "Images"

    n_train, n_test = _count_ppm(train_root), _count_ppm(test_root)
    print(f"[prepare] found {n_train} training images, {n_test} test images")

    if n_train != EXPECTED_TRAIN_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_TRAIN_IMAGES} training images, found {n_train}. "
            "Extraction looks incomplete -- rerun with --force-download."
        )
    if n_test != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"expected {EXPECTED_TEST_IMAGES} test images, found {n_test}. "
            "Extraction looks incomplete -- rerun with --force-download."
        )

    sentinel.write_text(f"train={n_train}\ntest={n_test}\n", encoding="utf-8")
    return extract_dir
