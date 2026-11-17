"""Fetch openly licensed candidate photos from Wikimedia Commons with credits.

Writes ROOT/gallery/candidates/{group}/{n}.jpg and candidates.json with author,
license and file-page URL. Selection among candidates happens later by
classifier confidence and visual review.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path.home() / "main/data/kpsa-demo/gallery"
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "kpsa-demo-gallery/0.1 (research demo; contact via axym.org)"}
OK_LICENSES = re.compile(r"^(CC BY(-SA)? [0-9.]+|CC0|Public domain|PD)", re.I)

QUERIES = {
    "Cats": ["tabby cat portrait", "siamese cat", "black cat sitting", "orange cat"],
    "Dogs": ["golden retriever dog", "border collie", "beagle dog", "german shepherd dog"],
    "Birds": ["european robin bird", "bald eagle", "flamingo", "macaw parrot"],
    "Cars": ["sports car", "vintage car", "convertible car", "taxi cab yellow"],
    "Insects": ["monarch butterfly", "ladybug", "honey bee flower", "dragonfly"],
    "Food": ["pizza", "cheeseburger", "strawberries", "espresso cup"],
    "More": ["lighthouse", "acoustic guitar", "sailboat", "mushroom forest", "volcano",
             "steam locomotive", "teddy bear", "red fox", "goldfish", "pineapple",
             "umbrella", "hot air balloon"],
}


def api(params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def strip(value):
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def search(query, limit=12):
    data = api({
        "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime", "iiurlwidth": 640,
    })
    out = []
    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        lic = strip(meta.get("LicenseShortName", {}).get("value"))
        if info.get("mime") != "image/jpeg" or not OK_LICENSES.match(lic):
            continue
        if info.get("width", 0) < 640 or info.get("height", 0) < 480:
            continue
        out.append({
            "title": page["title"], "thumb": info.get("thumburl"), "page": info.get("descriptionurl"),
            "author": strip(meta.get("Artist", {}).get("value")) or "Unknown",
            "license": lic, "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        })
    return out


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    existing = ROOT / "candidates.json"
    records = json.loads(existing.read_text()) if existing.exists() else []
    done = {r["query"] for r in records}
    for group, queries in QUERIES.items():
        for query in queries:
            if query in done:
                continue
            time.sleep(3)
            try:
                hits = search(query)
            except Exception as exc:  # network hiccup: skip query
                print("fail", query, exc)
                continue
            for n, hit in enumerate(hits[:6]):
                path = ROOT / "candidates" / group / f"{re.sub(r'[^a-z0-9]+', '-', query)}-{n}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    req = urllib.request.Request(hit["thumb"], headers=UA)
                    path.write_bytes(urllib.request.urlopen(req, timeout=30).read())
                except Exception as exc:
                    print("download fail", hit["title"], exc)
                    continue
                records.append({**hit, "group": group, "query": query, "path": str(path)})
                time.sleep(1.0)
            print(group, query, len(hits), flush=True)
    (ROOT / "candidates.json").write_text(json.dumps(records, indent=1))
    print("total", len(records))


if __name__ == "__main__":
    main()
