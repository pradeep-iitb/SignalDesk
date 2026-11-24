"""
Collect the Signal Desk dataset from the arXiv API (titles + abstracts, label = primary category).

    python collect_data.py

Writes
  data/arxiv_main.csv     papers submitted in 2025       -> train / validation / test
  data/arxiv_shifted.csv  papers submitted in 2012-2014  -> stress test (same classes, different time period)

Source: arXiv API (https://info.arxiv.org/help/api/index.html), queried politely (one request every
3 seconds, as the API terms of use ask). Only the arXiv id, date, title, abstract and category are
kept: no author names or any other personal data. Abstracts are used for research/education; the
metadata is CC0 (https://info.arxiv.org/help/api/tou.html).
"""
from __future__ import annotations

import csv
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = {  # arXiv primary category -> readable label
    "cs.CL": "nlp",
    "cs.CV": "vision",
    "cs.CR": "security",
    "cs.RO": "robotics",
    "cs.NI": "networking",
}
MAIN_WINDOWS = [("202501010000", "202503312359"), ("202504010000", "202506302359"),
                ("202507010000", "202509302359"), ("202510010000", "202512312359")]
SHIFT_WINDOWS = [("201201010000", "201412312359")]
MAIN_PER_WINDOW, MAIN_CAP = 250, 520      # request size per quarter, max kept per class
SHIFT_PER_WINDOW, SHIFT_CAP = 250, 60
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
OUT = Path(__file__).resolve().parent / "data"


def fetch(cat: str, start: str, end: str, n: int, retries: int = 4) -> list[dict]:
    q = f"cat:{cat} AND submittedDate:[{start} TO {end}]"
    url = API + "?" + urllib.parse.urlencode({"search_query": q, "start": 0, "max_results": n,
                                              "sortBy": "submittedDate", "sortOrder": "descending"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                root = ET.fromstring(r.read())
            entries = root.findall("a:entry", NS)
            if entries:
                break
        except Exception as e:  # network hiccup: wait and retry
            print(f"   retry {attempt + 1} ({e})")
        time.sleep(5 * (attempt + 1))
    else:
        return []
    rows = []
    for e in entries:
        prim = e.find("arxiv:primary_category", NS).attrib["term"]
        rows.append({
            "arxiv_id": e.find("a:id", NS).text.rsplit("/abs/", 1)[-1].split("v")[0],
            "published": e.find("a:published", NS).text[:10],
            "primary_category": prim,
            "title": " ".join(e.find("a:title", NS).text.split()),
            "abstract": " ".join(e.find("a:summary", NS).text.split()),
        })
    return rows


def collect(windows, per_window, cap) -> list[dict]:
    kept: dict[str, dict] = {}
    for cat, label in CLASSES.items():
        got = 0
        for start, end in windows:
            rows = fetch(cat, start, end, per_window)
            time.sleep(3)
            prim = [r for r in rows if r["primary_category"] == cat]  # cross-lists are dropped
            print(f"  {cat} {start[:6]}-{end[:6]}: {len(rows)} returned, {len(prim)} with this primary category")
            for r in prim[: max(0, cap // len(windows) + 1)]:
                if r["arxiv_id"] not in kept and got < cap:
                    kept[r["arxiv_id"]] = {**r, "label": label}
                    got += 1
        print(f"  -> {label}: {got}")
    return list(kept.values())


def save(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arxiv_id", "published", "primary_category", "label", "title", "abstract"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows -> {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    print("main set (2025):")
    save(collect(MAIN_WINDOWS, MAIN_PER_WINDOW, MAIN_CAP), OUT / "arxiv_main.csv")
    print("shifted set (2012-2014):")
    save(collect(SHIFT_WINDOWS, SHIFT_PER_WINDOW, SHIFT_CAP), OUT / "arxiv_shifted.csv")
