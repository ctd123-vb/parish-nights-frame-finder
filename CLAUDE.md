# Parish Nights Frame Finder

Pipeline: `find_frames.py` (extract + local scoring) -> `review.py` (creative review) -> `gallery.py` (approve).
See README.md for flags and output layout.

## When asked to review the shortlist (step 8)

1. `python review.py todo` lists thumbnails without a review (add `--video <name>` for one video).
2. Look at each thumbnail. Score it 1-10 on the fields in `RUBRIC` in `review.py`, pick tags from
   `TAGS`, set `room_sparse` and `best_crop`, and write one short `note`.
3. Judge the photo, never the people: no ratings of attractiveness, bodies, or appearance.
4. Write scores to `output/<video>/review.json` in the format in `review.py`'s docstring. Merge
   with existing entries; don't overwrite other frames' reviews.
5. Run `python review.py merge` to rebuild `output/results.csv`.
