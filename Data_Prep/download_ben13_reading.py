"""
Ben13's *reading* corpus — real Torah chanting, already segmented.
==================================================================
This is a different section of ben13.co.il from the one this project has been
using. Everything before now came from `teamimtora`/`teamim-meg`/`teamim-haf`:
75 clips of the teacher singing the *names* of the ta'amim. Useful as
reference, but they had to be cut apart with CTC, and that is where nearly
every data bug in this project came from -- phantom words, truncated
melismas, a text that disagreed with its own audio.

The reading section is a different thing entirely:

  * 52,226 rows covering the whole Torah, not 75 clips
  * actual chanted verses, not ta'am names
  * **already segmented** -- one mp3 per phrase, with its own pointed text
  * four traditions (yerushalmi / maroko / ashkenazi / bavly)

The segmentation is not arbitrary: Ben13 cuts each verse at its disjunctive
accents, which is exactly the structure Data_Prep/taam_grammar.py derives
independently. Genesis 1:1 splits after אֱלֹהִ֑ים -- the etnachta -- in both.
That agreement is what makes this corpus checkable rather than merely
convenient.

DATA SOURCE
-----------
The row table ships inside the site's own JS bundle
(assets/table_maintext-*.js) as a JSON array. Audio lives at
  https://www.ben13.co.il/audio-files/<nusach>/<url lower-cased>
Fields per row: id, book, perek, pasuk, block, aliya, parasha, text, url.
`block` -1 and 0 are headings (aliyah title, verse number) -- not audio of
chanted text -- and are skipped.

Downloads are rate-limited and resumable: an already-present file is never
re-fetched, so an interrupted run costs nothing.

Usage:
  python Data_Prep/download_ben13_reading.py --perek 1 --dry-run
  python Data_Prep/download_ben13_reading.py --perek 1
  python Data_Prep/download_ben13_reading.py --all
"""
import argparse
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The site's certificate chain is missing an Authority Key Identifier, which
# Python's default verifier rejects but the system trust store accepts. Same
# fix already used in webapp/server.py and for the Text-Fabric download.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLE = PROJECT_DIR / "data" / "ben13_reading" / "maintext.json"
OUT_DIR = PROJECT_DIR / "data" / "ben13_reading"
BASE = "https://www.ben13.co.il/audio-files"
NUSACH = "yerushalmi"          # the tradition this project targets
UA = "Mozilla/5.0 (research; local use)"

# Downloads run in parallel. The first version used a single connection with
# a 0.35s pause, which made 42,782 files a ~4.2 hour job of almost pure
# waiting -- the delay, not the bandwidth, was the entire bottleneck.
# WORKERS is deliberately modest: a browser opens ~6 connections per host as
# a matter of course, so this stays in ordinary-client territory rather than
# hammering someone else's server.
WORKERS = 10


def load_rows():
    with open(TABLE, encoding="utf-8") as f:
        return json.load(f)


def audio_url(row: str) -> str:
    # the site lower-cases the path before requesting it (see bi()/yi() in
    # the bundle); the only exception is a SHENI-HAMISHI directory, which is
    # not used inside the Torah reading rows
    return f"{BASE}/{NUSACH}/{row.lstrip('/').lower()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="1", help="1=Genesis")
    ap.add_argument("--perek", default=None, help="chapter; omit for all")
    ap.add_argument("--parasha", default=None)
    ap.add_argument("--all", action="store_true", help="whole Torah")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows()
    sel = rows
    if not args.all:
        sel = [r for r in sel if r["book"] == args.book]
        if args.perek:
            sel = [r for r in sel if r["perek"] == args.perek]
        if args.parasha:
            sel = [r for r in sel if r["parasha"] == args.parasha]

    # block -1/0 are headings (aliyah label, verse number), not chanted text
    segs = [r for r in sel if r["block"] not in ("-1", "0") and r.get("url")]
    if args.limit:
        segs = segs[: args.limit]

    print(f"{len(segs):,} audio segments selected "
          f"(of {len(sel):,} rows scanned)")
    if args.dry_run:
        for r in segs[:12]:
            print(f"  {r['book']}/{r['perek']}:{r['pasuk']} b{r['block']:<3} "
                  f"{r['text'][:44]}")
            print(f"      {audio_url(r['url'])}")
        return

    manifest, counts = [], {"ok": 0, "skipped": 0, "failed": 0, "done": 0}
    lock = threading.Lock()

    def fetch(r):
        rel = r["url"].lstrip("/").lower()
        dest = OUT_DIR / NUSACH / rel
        entry = {
            "id": r["id"], "book": r["book"], "perek": r["perek"],
            "pasuk": r["pasuk"], "block": r["block"], "aliya": r["aliya"],
            "parasha": r["parasha"], "text": r["text"],
            "source_url": audio_url(r["url"]),
            "local_path": str(dest.relative_to(PROJECT_DIR)),
        }
        if dest.exists() and dest.stat().st_size > 0:
            return "skipped", entry
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(audio_url(r["url"]),
                                         headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            # Write via a temp name so an interrupted run never leaves a
            # half-written mp3 that resume logic would treat as complete.
            # The temp name carries the thread id: several manifest rows can
            # point at the same url, and with a shared ".part" name two
            # workers raced -- one renamed it, the other then failed with
            # "no such file". That accounted for most of the 5,124 failures
            # in the first full run.
            tmp = dest.with_suffix(dest.suffix + f".{threading.get_ident()}.part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return "ok", entry
        except Exception as e:
            return f"failed:{e}", entry

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch, r): r for r in segs}
        for fut in as_completed(futures):
            status, entry = fut.result()
            with lock:
                counts["done"] += 1
                if status == "ok":
                    counts["ok"] += 1
                    manifest.append(entry)
                elif status == "skipped":
                    counts["skipped"] += 1
                    manifest.append(entry)
                else:
                    counts["failed"] += 1
                    if counts["failed"] <= 20:
                        print(f"  FAIL {entry['local_path']}: {status[7:]}")
                if counts["done"] % 500 == 0:
                    print(f"  {counts['done']}/{len(segs)}  (new {counts['ok']}, "
                          f"cached {counts['skipped']}, failed {counts['failed']})",
                          flush=True)

    ok, skipped, failed = counts["ok"], counts["skipped"], counts["failed"]

    mpath = OUT_DIR / f"manifest_{NUSACH}.json"
    existing = []
    if mpath.exists():
        with open(mpath, encoding="utf-8") as f:
            existing = json.load(f)
    seen = {e["id"] for e in manifest}
    merged = manifest + [e for e in existing if e["id"] not in seen]
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)

    print(f"\ndownloaded {ok}, already had {skipped}, failed {failed}")
    print(f"manifest: {len(merged):,} entries -> {mpath.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
