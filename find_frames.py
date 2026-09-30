#!/usr/bin/env python3
"""Pull the best still frames from dance videos.

Usage:
    python find_frames.py input/                 # every video in a folder (skips ones already done)
    python find_frames.py input/clip.mov         # one video
    python find_frames.py input/ --workers 6 --cleanup all

Per video, writes output/<video>/:
    frame_scores.csv   every extracted frame, its metrics, and which filter dropped it
    shortlist/         top frames at full resolution
    thumbs/            smaller copies for the gallery and review
    shortlist.csv      the shortlisted frames
    report.json        specs, filter counts, quality notes
Then rebuilds output/results.csv and output/summary.csv across all videos.
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np

from framefinder.results import build_results, write_csv
from framefinder.scoring import FrameScore, Scorer, Thresholds, group_moments, pick_shortlist
from framefinder.video import extract_frames, frame_time, list_videos, probe

_scorer = None


def _init_worker(th_dict):
    global _scorer
    _scorer = Scorer(Thresholds(**th_dict))


def _score_one(args):
    path, t = args
    return _scorer.score(Path(path), t)


def score_frames(frames, fps, th: Thresholds, workers: int) -> list[FrameScore]:
    jobs = [(str(p), frame_time(p, fps)) for p in frames]
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(asdict(th),)) as ex:
        for i, r in enumerate(ex.map(_score_one, jobs, chunksize=4), 1):
            results.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f"\r  scored {i}/{len(jobs)}", end="", flush=True)
    print()
    return results


def funnel(scores: list[FrameScore]) -> dict:
    n = len(scores)
    order = ["no_clear_face", "bad_exposure", "blurry_face", "eyes_closed"]
    dropped = {k: sum(s.fail == k for s in scores) for k in order}
    left, out = n, {"extracted": n}
    for k, label in zip(order, ["has_clear_face", "exposure_ok", "face_sharp", "eyes_open"]):
        left -= dropped[k]
        out[label] = left
    out["unreadable"] = sum(s.fail == "unreadable" for s in scores)
    return out


def quality_notes(specs, scores: list[FrameScore], th: Thresholds) -> list[str]:
    notes = []
    faced = [s for s in scores if s.n_clear > 0 and s.face_sharpness > 0]
    if specs.hdr:
        notes.append(f"Video is HDR ({specs.hdr_format}); frames were tone mapped to SDR (mobius curve, 203-nit white).")
    if faced:
        blur = sum(s.fail == "blurry_face" for s in faced) / len(faced)
        med = float(np.median([s.face_sharpness for s in faced]))
        notes.append(f"Median face sharpness {med:.0f} (blur cutoff {th.min_sharpness:.0f}); "
                     f"{blur:.0%} of frames with faces were too blurry.")
        if blur > 0.4:
            notes.append("Heavy motion blur. For dancing, shoot 60 fps on iPhone (forces a faster shutter) "
                         "or 1/250s+ on the mirrorless, and add light if you can.")
        med_frac = float(np.median([s.face_frac for s in faced]))
        if med_frac < 0.08:
            notes.append(f"Faces are small (median {med_frac:.0%} of frame height). Get closer or zoom in.")
        dark = sum(s.face_luma < th.min_face_luma for s in faced) / len(faced)
        if dark > 0.25:
            notes.append(f"{dark:.0%} of face frames are underexposed. The room lighting is dim on faces.")
        clipped = sum(s.clipped > th.max_clipped for s in scores) / max(1, len(scores))
        if clipped > 0.1:
            notes.append(f"{clipped:.0%} of frames have large blown-out areas (lights or windows).")
    face_rate = len(faced) / max(1, len(scores))
    if face_rate < 0.3:
        notes.append(f"Only {face_rate:.0%} of frames had a usable face. Camera may be too far or facing backs.")
    px = specs.width * specs.height
    if specs.bitrate_mbps and specs.bitrate_mbps < (8 if px <= 1920 * 1080 else 20):
        notes.append(f"Low bitrate ({specs.bitrate_mbps:.1f} Mbps). Expect compression artifacts in stills.")
    if specs.fps < 50:
        notes.append(f"Shot at {specs.fps:.0f} fps. 60 fps gives sharper stills of fast moves.")
    return notes


def save_thumb(src: Path, dst: Path, long_side: int = 1400):
    img = cv2.imread(str(src))
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def process_video(path: Path, out_root: Path, args, th: Thresholds) -> dict | None:
    vdir = out_root / path.stem
    if (vdir / "report.json").exists() and not args.force:
        print(f"= {path.name}: already done, skipping (use --force to redo)")
        return json.loads((vdir / "report.json").read_text())
    t0 = time.time()
    print(f"\n== {path.name}")
    specs = probe(path)
    print(specs.summary())

    frames_dir = vdir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    print(f"  extracting at {args.fps} fps{' with HDR->SDR tone mapping' if specs.hdr else ''}...")
    frames = extract_frames(specs, frames_dir, fps=args.fps, white_nits=args.hdr_white)
    print(f"  {len(frames)} frames extracted")

    scores = score_frames(frames, args.fps, th, args.workers)
    cols = [f.name for f in fields(FrameScore)]
    write_csv(vdir / "frame_scores.csv", [s.row() for s in scores], cols)

    passed = [s for s in scores if not s.fail]
    moments = group_moments(passed)
    best = [max(g, key=lambda s: s.local_score) for g in moments]
    shortlist = pick_shortlist(best, n=args.top)

    for d in ("shortlist", "thumbs"):
        if (vdir / d).exists():
            shutil.rmtree(vdir / d)
        (vdir / d).mkdir(parents=True)
    rows = []
    for s in shortlist:
        name = f"{path.stem}_t{s.t:07.2f}.jpg"
        shutil.copy2(frames_dir / s.file, vdir / "shortlist" / name)
        save_thumb(frames_dir / s.file, vdir / "thumbs" / name)
        rows.append({**s.row(), "file": name, "source_frame": s.file})
    write_csv(vdir / "shortlist.csv", rows, ["file", "source_frame"] + [c for c in cols if c != "file"])

    counts = funnel(scores)
    counts["moments"] = len(moments)
    counts["shortlisted"] = len(shortlist)
    report = {
        "video": path.name, "specs": specs.to_dict(), "counts": counts,
        "notes": quality_notes(specs, scores, th), "thresholds": asdict(th),
        "seconds": round(time.time() - t0, 1),
    }
    (vdir / "report.json").write_text(json.dumps(report, indent=2))

    if args.cleanup == "all":
        shutil.rmtree(frames_dir)
    elif args.cleanup == "rejects":
        for s in scores:
            if s.fail:
                (frames_dir / s.file).unlink(missing_ok=True)

    print("  filter funnel: " + " -> ".join(f"{k} {v}" for k, v in counts.items() if k != "unreadable"))
    for n in report["notes"]:
        print(f"  note: {n}")
    print(f"  done in {report['seconds']}s")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="a video file or a folder of videos")
    ap.add_argument("--output", type=Path, default=Path("output"))
    ap.add_argument("--fps", type=float, default=8.0, help="frames per second to extract (default 8)")
    ap.add_argument("--top", type=int, default=15, help="shortlist size per video (default 15)")
    ap.add_argument("--workers", type=int, default=4, help="parallel scoring processes (default 4)")
    ap.add_argument("--cleanup", choices=["none", "rejects", "all"], default="rejects",
                    help="delete extracted frames after scoring: none, only filtered-out ones (default), "
                         "or all (keeps only the shortlist; best for big batches)")
    ap.add_argument("--hdr-white", type=int, default=203,
                    help="HDR nits mapped to SDR white (default 203). Lower = brighter frames.")
    ap.add_argument("--min-sharpness", type=float, default=Thresholds.min_sharpness)
    ap.add_argument("--force", action="store_true", help="re-process videos that are already done")
    args = ap.parse_args()

    th = Thresholds(min_sharpness=args.min_sharpness)
    videos = list_videos(args.target)
    if not videos:
        sys.exit(f"No videos found at {args.target}")
    args.output.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}]", end="")
        try:
            r = process_video(v, args.output, args, th)
        except Exception as e:  # keep the batch going; one bad file shouldn't stop 200
            print(f"  ERROR on {v.name}: {e}")
            summary.append({"video": v.name, "error": str(e)})
            continue
        sp = r["specs"]
        summary.append({"video": r["video"], "resolution": f"{sp['width']}x{sp['height']}", "fps": round(sp["fps"], 2),
                        "hdr": sp["hdr_format"] or "SDR", **r["counts"], "notes": " | ".join(r["notes"])})

    cols = ["video", "resolution", "fps", "hdr", "extracted", "has_clear_face", "exposure_ok", "face_sharp",
            "eyes_open", "moments", "shortlisted", "notes", "error"]
    write_csv(args.output / "summary.csv", summary, cols)
    rows = build_results(args.output)
    print(f"\nWrote {args.output}/summary.csv and {args.output}/results.csv ({len(rows)} shortlisted frames)")
    print("Next: review the shortlist (see README), then run: python gallery.py")


if __name__ == "__main__":
    main()
