"""
Download Ben13's teaching clips (ben13.co.il) as golden reference audio.
=========================================================================
Ben13 is a Sephardic-Jerusalem (yerushalmi) ta'amim teaching site: short,
professionally produced, isolated demo clips (~4s each) for Torah, Megillot,
and Haftara cantillation. Not a training-data source (one voice, isolated
demos, not continuous natural reading) — this is reference/validation
material for the practice app's "golden teacher clip" feature and for
sanity-checking the rarest classes (Shalshelet, Yerach Ben Yomo, Qarney
Para), which the real training corpus has only 1-2 instances of.

The site is a JS single-page app; its manifest (id/order/text/textSfaradi/
url/type) lives inline in the deployed JS bundle, not behind a normal API.
The bundle's hashed filename changes on deploy, so it's discovered from the
homepage each run rather than hardcoded. The manifest is JS-literal, not
strict JSON (it contains raw \\xNN escapes for niqqud spacing), so it's
parsed with a targeted regex rather than json.loads.

Each entry's Hebrew `text` field embeds the actual cantillation-mark Unicode
diacritics (e.g. "זקף-גד֕ול" contains the Zaqef Gadol codepoint itself), so
it decodes straight into this project's internal class names via
clean_align_v2.py's TAAM_MAP — no separate Hebrew-string lookup needed.

Usage:
  python Data_Prep/download_ben13.py --dry-run     # show the decoded manifest
  python Data_Prep/download_ben13.py               # download everything
  python Data_Prep/download_ben13.py --types teamimtora
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_align_v2 import TAAM_MAP  # noqa: E402

# This environment's outbound traffic goes through an inspecting proxy that
# presents its own CA — trusted by the system store (curl/yt-dlp work fine)
# but not by requests' bundled certifi store. Same fix as yt-dlp's
# --no-check-certificates elsewhere in Data_Prep.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "data" / "ben13"
BASE = "https://www.ben13.co.il"

TYPE_NAMES = {"1": "teamimtora", "2": "teamim-meg", "3": "teamim-haf"}

ENTRY_RE = re.compile(
    r'"id":"(\d+)","order":"(\d+)".*?'
    r'"text":"(.*?)","textSfaradi":"(.*?)","url":"(.*?)","type":"(\d+)"'
)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; taamify-research/1.0)"}


def find_bundle_url() -> str:
    r = requests.get(BASE + "/", headers=HEADERS, timeout=20, verify=False)
    r.raise_for_status()
    m = re.search(r'<script[^>]*type="module"[^>]*src="([^"]+)"', r.text)
    if not m:
        raise SystemExit("could not find the JS bundle <script> tag in the homepage")
    src = m.group(1)
    return src if src.startswith("http") else BASE + src


def fetch_manifest() -> list:
    bundle_url = find_bundle_url()
    print(f"  bundle: {bundle_url}")
    r = requests.get(bundle_url, headers=HEADERS, timeout=30, verify=False)
    r.raise_for_status()
    js = r.text

    entries = []
    for m in ENTRY_RE.finditer(js):
        id_, order, text, text_sfaradi, url, type_ = m.groups()
        entries.append({
            "id": id_,
            "order": int(order),
            "text": text,
            "textSfaradi": text_sfaradi,
            "url": url,
            "type": type_,
        })
    if not entries:
        raise SystemExit("manifest regex matched nothing — the bundle's format "
                          "may have changed; inspect it by hand")
    return entries


def decode_taam_sequence(text: str) -> list:
    """Ordered internal class names from the Unicode cantillation diacritics
    embedded in a Hebrew label, e.g. 'זקף-גד֕ול' -> ['Zaqef_Gadol']."""
    return [TAAM_MAP[ch] for ch in text if ch in TAAM_MAP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="print the decoded manifest, download nothing")
    ap.add_argument("--types", nargs="+", default=list(TYPE_NAMES.values()),
                     choices=list(TYPE_NAMES.values()),
                     help="which clip types to fetch (default: all three)")
    args = ap.parse_args()

    print("Fetching Ben13 manifest...")
    entries = fetch_manifest()
    by_type = {}
    for e in entries:
        by_type.setdefault(TYPE_NAMES[e["type"]], []).append(e)

    for type_name in args.types:
        group = sorted(by_type.get(type_name, []), key=lambda e: e["order"])
        print(f"\n{'=' * 70}\n  {type_name}: {len(group)} clips\n{'=' * 70}")

        manifest_out = []
        out_dir = OUT_DIR / type_name
        for e in group:
            seq = decode_taam_sequence(e["text"])
            url_path = e["url"].lower()
            full_url = f"{BASE}/audio-files/yerushalmi/{url_path}"
            local_name = Path(url_path).name
            local_path = out_dir / local_name
            manifest_out.append({
                **e,
                "taam_sequence": seq,
                "resolved_url": full_url,
                "local_path": str(local_path.relative_to(PROJECT_DIR)),
            })
            flag = "" if seq else "  <- NO TA'AM DECODED, check by hand"
            print(f"    {e['order']:>3} {e['text'][:35]:<35} "
                  f"-> {','.join(seq) or '(none)'}{flag}")

        if args.dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        for item in manifest_out:
            local_path = PROJECT_DIR / item["local_path"]
            if local_path.exists():
                continue
            r = requests.get(item["resolved_url"], headers=HEADERS, timeout=30, verify=False)
            if r.status_code != 200:
                print(f"    ERROR {r.status_code}: {item['resolved_url']}")
                continue
            local_path.write_bytes(r.content)
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_out, f, ensure_ascii=False, indent=1)
        print(f"  wrote {len(manifest_out)} clips + manifest -> "
              f"{manifest_path.relative_to(PROJECT_DIR)}")

    if args.dry_run:
        print("\n--dry-run: nothing downloaded")


if __name__ == "__main__":
    main()
