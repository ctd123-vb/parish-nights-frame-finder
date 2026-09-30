"""Shared CSV/JSON helpers: per-video shortlist + optional review -> one results.csv."""
import csv
import json
from pathlib import Path

REVIEW_FIELDS = ["sharp_faces", "real_smiles", "energy", "young_fun", "room_full", "headline_space", "crop_fit"]
RESULT_COLUMNS = [
    "id", "video", "file", "t", "timestamp", "final_score", "local_score", "review_avg",
    *REVIEW_FIELDS, "tags", "room_sparse", "best_crop", "review_note",
    "n_clear", "face_frac", "face_sharpness", "min_sharpness", "smile", "eyes_open",
    "face_luma", "clipped", "width", "height", "backup", "decision",
]


def fmt_ts(t: float) -> str:
    m, s = divmod(float(t), 60)
    return f"{int(m)}:{s:05.2f}"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def final_score(local: float, review_avg: float | None) -> float:
    """Local score is 0-100; review is 1-10 per category. Review counts more once it exists."""
    if review_avg is None:
        return round(local, 1)
    return round(0.35 * local + 0.65 * review_avg * 10, 1)


def build_results(output_dir: Path) -> list[dict]:
    """Merge every video's shortlist.csv + review.json + gallery decisions into output/results.csv."""
    decisions = load_json(output_dir / "decisions.json", {})
    rows = []
    for vdir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        shortlist = read_csv(vdir / "shortlist.csv")
        if not shortlist:
            continue
        review = load_json(vdir / "review.json", {})
        for r in shortlist:
            rid = f"{vdir.name}/{r['file']}"
            rv = review.get(r["file"], {})
            scores = [float(rv[k]) for k in REVIEW_FIELDS if rv.get(k) not in (None, "")]
            ravg = round(sum(scores) / len(scores), 2) if scores else None
            rows.append({
                **r, **{k: rv.get(k, "") for k in REVIEW_FIELDS},
                "id": rid, "video": vdir.name, "timestamp": fmt_ts(r["t"]),
                "review_avg": "" if ravg is None else ravg,
                "final_score": final_score(float(r["local_score"]), ravg),
                "tags": "|".join(rv.get("tags", [])),
                "room_sparse": rv.get("room_sparse", ""),
                "best_crop": rv.get("best_crop", ""),
                "review_note": rv.get("note", ""),
                "decision": decisions.get(rid, ""),
            })
    # reviewed frames first (their scores include the review), then by score
    rows.sort(key=lambda r: (r["review_avg"] == "", -float(r["final_score"])))
    write_csv(output_dir / "results.csv", rows, RESULT_COLUMNS)
    return rows
