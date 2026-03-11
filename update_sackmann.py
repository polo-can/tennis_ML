"""Download/update ATP match data from Jeff Sackmann's tennis_atp GitHub repo."""

import urllib.request
import os
from pathlib import Path

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

FILES = [
    # Match files
    *[f"atp_matches_{year}.csv" for year in range(2015, 2026)],
    # Reference files
    "atp_players.csv",
    "atp_rankings_current.csv",
]


def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    dest = DATA_DIR / filename
    try:
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        print(f"  {filename}: {size:,} bytes")
        return True
    except Exception as e:
        print(f"  {filename}: SKIPPED ({e})")
        return False


def main():
    print("Downloading latest ATP data from JeffSackmann/tennis_atp...\n")

    success = 0
    for f in FILES:
        if download_file(f):
            success += 1

    print(f"\nDone: {success}/{len(FILES)} files downloaded to {DATA_DIR}/")


if __name__ == "__main__":
    main()
