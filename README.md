# Parish Nights Frame Finder

Pulls the best still frames from dance videos for the Parish Nights photo library
(Instagram posts, Meta ads, digital flyers). Runs locally. Works on one video or a folder of 200.

## Setup (once)

```bash
./setup.sh                    # checks/installs ffmpeg + Python, makes .venv, fetches face models
source .venv/bin/activate
```

## Run

```bash
# 1. Drop videos in input/, then:
python find_frames.py input/

# 2. Creative review (step 8). Either ask Claude Code "review the shortlist", or automate it:
export ANTHROPIC_API_KEY=...
python review.py ai

# 3. Approve in the browser
python gallery.py             # open http://localhost:8765
```

Gallery keys: **K** keep, **J** reject, **U** undo, **left/right** move, **F** open full resolution.
Keep copies the full-res frame to `keepers/`. Reject moves it to `output/<video>/rejected/`.

## What it does per video

1. Reads specs with ffprobe: resolution, fps, orientation (rotation applied), codec, bit depth, HDR type.
2. Extracts 8 frames/sec at full resolution to high-quality JPEG (`-q:v 2`).
   HDR (HLG, PQ, Dolby Vision) is tone mapped to SDR: `zscale` + `tonemap=mobius`, 203-nit reference
   white. Mobius keeps skin tones close to the original and only compresses highlights.
   The more common `hable` curve darkened test frames by about a third, which is the "washed out" look.
3. Scores every frame locally (MediaPipe, CPU only):
   - faces: full-frame + 2x2 tiled detection, so small faces in wide shots are found
   - face sharpness: Laplacian variance on each face, resized to 128px and contrast-normalized
   - eyes open and smile: MediaPipe face blendshapes (`eyeBlink*`, `mouthSmile*`)
   - exposure: face brightness, frame brightness, blown-out area
4. Drops frames in this order: no clear face, bad exposure, blurry face, eyes closed.
5. Groups near-duplicates into "moments" (image hash + face layout) and keeps the best of each.
6. Shortlists the top 15, at least 1 second apart.

## Output

```
output/
  results.csv          every shortlisted frame, all videos: scores, tags, decision (sorted by score)
  summary.csv          one row per video: specs, filter counts, quality notes
  decisions.json       gallery keep/reject decisions
  <video>/
    frame_scores.csv   every extracted frame, its metrics, and the filter that dropped it
    shortlist/         full-res shortlisted frames
    thumbs/            1400px copies for the gallery and review
    review.json        1-10 creative scores + tags
    report.json        specs, counts, notes
keepers/               approved frames, full resolution
```

`final_score` = 35% local score (0-100) + 65% review average (1-10, scaled). Unreviewed frames
use the local score only and sort after reviewed ones.

## Batch tips (100-200 videos)

- Re-running skips videos that are already done. `--force` redoes them.
- One bad file won't stop the batch; errors go to `summary.csv`.
- Disk: a 1-minute 4K clip is roughly 480 frames and 1-2 GB of JPEGs. The default
  `--cleanup rejects` deletes frames that failed a filter. `--cleanup all` keeps only the shortlist.
- Speed: about 1.25x the clip length for 4K 60fps HDR on a 4-core machine. Raise `--workers` on
  a bigger machine.
- `review.py ai` saves after every frame, so you can stop and resume. It uses Claude Opus 5.5
  (`--model` to change) with low effort, on the 1400px thumbnails. Server-side fallback is on,
  so a declined request is retried on another model instead of failing.

## Tuning

| Flag | Default | Effect |
|---|---|---|
| `--fps` | 8 | frames extracted per second |
| `--top` | 15 | shortlist size per video |
| `--min-sharpness` | 15 | blur cutoff. Sharp faces score ~30+, motion blur under 10. Check `frame_scores.csv` |
| `--hdr-white` | 203 | lower (e.g. 150) = brighter HDR frames, highlights clip sooner |
| `--workers` | 4 | parallel scoring processes |
| `--cleanup` | rejects | `none`, `rejects`, or `all` |

Other thresholds (min face size, blink cutoff, exposure limits) are in `framefinder/scoring.py`
(`Thresholds`).

## Browser version

`web/frame-finder.html` is the same pipeline as a single web page, published as a Claude artifact:
pick a video, it scans in the browser (nothing uploads), Claude scores the shortlist, press K/J,
download keepers as a zip. It needs these files next to it when published: `lib/jszip.min.js`
(npm `jszip`), `mp/vision_bundle.mjs` + `mp/vision_wasm_internal.{js,wasm}` (npm
`@mediapipe/tasks-vision`), and `mp/models.js` (the two files in `models/`, base64-encoded as
`self.PN_MODELS = {det, lm}`).
