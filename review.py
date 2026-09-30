#!/usr/bin/env python3
"""Step 8: creative review of each shortlisted frame (1-10 scores + tags).

Two ways to fill in output/<video>/review.json:

  1. Ask Claude Code to do it ("review the shortlist"). It runs `python review.py todo`, looks at
     each thumbnail, and writes review.json in the format below.
  2. Automated, for big batches, with the Claude API (needs ANTHROPIC_API_KEY):
         python review.py ai                # every video missing a review
         python review.py ai --video clip1  # one video

Then `python review.py merge` (the ai command does this for you) rebuilds output/results.csv.

review.json format, keyed by shortlist file name:
  {"clip1_t0012.50.jpg": {"sharp_faces": 8, "real_smiles": 7, "energy": 9, "young_fun": 8,
    "room_full": 6, "headline_space": 5, "crop_fit": 7, "room_sparse": false,
    "best_crop": "4:5", "tags": ["couple mid-move"], "note": "..."}}
"""
import argparse
import base64
import json
import sys
from pathlib import Path

from framefinder.results import REVIEW_FIELDS, build_results, load_json, read_csv

TAGS = ["couple mid-move", "smiling close-up", "crowd", "candid"]

RUBRIC = f"""You are reviewing a still frame pulled from a video of Parish Nights, a country swing \
dance night for young adults. The photo library feeds Instagram posts, Meta ads, and digital flyers.

Score 1-10 (10 = best) on:
- sharp_faces: faces are in focus and the moment is flattering (expression, angle, not mid-blink or mid-word)
- real_smiles: genuine, natural smiles rather than neutral or forced faces
- energy: movement, fun, a sense of a lively night
- young_fun: the scene reads as a young, fun night out (styling, vibe, activity)
- room_full: the room looks busy and well attended. Set room_sparse=true if it looks empty or thin
- headline_space: clean area (floor, wall, ceiling, dark background) where headline text could sit
- crop_fit: how well it survives a 4:5 or 9:16 crop without cutting key people. best_crop is \
"4:5", "9:16", "both", or "neither"

Tags (one or more): {", ".join(TAGS)}.

Judge the photo, not the people. Never rate anyone's attractiveness, body, or appearance. \
"Flattering" means focus, expression, and angle only. Keep note to one short sentence about \
what makes the frame useful or not."""

SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "integer", "description": "1-10"} for k in REVIEW_FIELDS},
        "room_sparse": {"type": "boolean"},
        "best_crop": {"type": "string", "enum": ["4:5", "9:16", "both", "neither"]},
        "tags": {"type": "array", "items": {"type": "string", "enum": TAGS}},
        "note": {"type": "string"},
    },
    "required": [*REVIEW_FIELDS, "room_sparse", "best_crop", "tags", "note"],
    "additionalProperties": False,
}


def videos(output: Path, only: str | None):
    for vdir in sorted(p for p in output.iterdir() if p.is_dir()):
        if (vdir / "shortlist.csv").exists() and (only is None or vdir.name == only):
            yield vdir


def cmd_todo(args):
    """List thumbnails that still need a review."""
    n = 0
    for vdir in videos(args.output, args.video):
        done = load_json(vdir / "review.json", {})
        for r in read_csv(vdir / "shortlist.csv"):
            if r["file"] not in done:
                print(vdir / "thumbs" / r["file"])
                n += 1
    print(f"{n} frames need review", file=sys.stderr)


def review_one(client, model: str, image: Path) -> dict | None:
    import anthropic
    data = base64.standard_b64encode(image.read_bytes()).decode()
    try:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=4000,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": RUBRIC},
            ]}],
        )
    except anthropic.RateLimitError:
        print("  rate limited even after retries; stopping so you can re-run later")
        raise
    except anthropic.APIStatusError as e:
        print(f"  API error on {image.name}: {e.status_code} {e.message}")
        return None
    if resp.stop_reason in ("refusal", "max_tokens"):
        print(f"  no review for {image.name} (stop_reason={resp.stop_reason})")
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)


def cmd_ai(args):
    import anthropic
    client = anthropic.Anthropic()
    for vdir in videos(args.output, args.video):
        path = vdir / "review.json"
        review = load_json(path, {})
        todo = [r["file"] for r in read_csv(vdir / "shortlist.csv") if r["file"] not in review]
        if not todo:
            continue
        print(f"{vdir.name}: reviewing {len(todo)} frames")
        for name in todo:
            result = review_one(client, args.model, vdir / "thumbs" / name)
            if result:
                review[name] = result
                path.write_text(json.dumps(review, indent=2))   # save as we go, so re-runs resume
                print(f"  {name}: avg {sum(result[k] for k in REVIEW_FIELDS) / len(REVIEW_FIELDS):.1f} "
                      f"{result['tags']}")
    cmd_merge(args)


def cmd_merge(args):
    rows = build_results(args.output)
    print(f"Wrote {args.output}/results.csv ({len(rows)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["todo", "ai", "merge"])
    ap.add_argument("--output", type=Path, default=Path("output"))
    ap.add_argument("--video", help="only this video (folder name under output/)")
    ap.add_argument("--model", default="claude-opus-5-5")
    args = ap.parse_args()
    {"todo": cmd_todo, "ai": cmd_ai, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    main()
