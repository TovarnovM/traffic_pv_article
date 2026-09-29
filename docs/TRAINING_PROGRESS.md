# Training progress data and figures

This separate review projection contains **9,165 accepted records: 15 histories × 611 iterations**, with 34 accepted segments. Each history ends at 10,010,624 accepted environment steps (`iteration × 16,384`). The 28 primary input files total 626,808,690 bytes. The curated JSONL is 1,886,128 bytes, or **204,228 bytes gzipped**. Source artifacts and the training pipeline were not modified.

The public inputs are `data/training/progress.jsonl.gz` and `data/training/manifest.json`. The manifest supplies the schema, exact original relative paths/SHA-256 hashes/sizes, accepted ranges, segment-to-source mapping, selected-checkpoint receipt bindings, and compact rejection decisions. `source_id` indexes its `sources` array; `source_row` is a one-based physical JSONL line. Each data row has family, seed, iteration, environment steps, segment, source ID/row, selected flag, and exactly two metrics. Source hashes attest original files; the public package validates their format and row mapping, but cannot rehash originals that it deliberately does not ship.

## Metrics and interpretation

- **PV speed:** `pv_speed_mean`, in **cells per simulation step**, directly preserves saved `v_pv_mean`. The callback records each completed episode's mean priority-vehicle speed, then RLlib reports a rolling episode mean pooled over sampled training conditions. This is not an equal-weight condition macro-average or a held-out score.
- **Team return:** `team_return_mean`, a **dimensionless episode reward sum**, preserves the custom metric of that name. The environment accumulates one team reward per physical step: `PV speed / 6 − 0.001 × AV applied lane changes − 0.002 × AV rejected lane changes`; the callback supplies its completed-episode total. Episodes have 1,000 measurement steps. This is not RLlib's summed multi-agent episode return, nor a claim about a particular policy's learner reward.

Exact field paths are arrays in `manifest.json`: CPU metrics are under `result → env_runners → custom_metrics`; J_G metrics are under top-level `custom_metrics`. CPU counters come from `counters.num_env_steps_sampled`; J_G counters from `num_env_steps`. Both common metrics are present and finite in every retained row. No missing metric is filled. Condition-specific summaries (which have missing entries) and learner diagnostics are omitted altogether. In particular, the finalized J_G logs have **no saved PPO loss or entropy series**.

JSON uses Python's round-trip float representation without decimal rounding. Compression uses gzip level9, empty filename and `mtime=0`. Public tests check exact float reserialization and same-runtime gzip replay.

## Accepted lineage

Seeds below are 20260914, 20260915, 20260916, in that order.

| Family | Selected iterations | Accepted segments |
|---|---|---|
| A | 500 / 600 / 450 | All seeds: 1–306; 307–611 |
| P_L | 550 / 500 / 500 | All seeds: 1–306; 307–611 |
| P_G | 500 / 450 / 450 | Seed14: 1–306; 307–575; 576–611. Others: 1–306; 307–611 |
| J_L | 450 / 550 / 450 | All seeds: 1–306; 307–611 |
| J_G | 450 / 500 / 450 | Seed14: 1–150; 151–175; 176–425; 426–611. Seed15: 1–150; 151–611. Seed16: 1–175; 176–200; 201–611 |

CPU sources are the inventory's collected histories, usually final attempt002 and its saved predecessor. P_L16 uses final attempt003 with fresh attempt002 as predecessor; failed attempt001 iterations1–105 are excluded entirely. P_G14 uses attempt003's `source/source/metrics.jsonl`1–306, `source/partial/metrics.jsonl`307–575, and final `metrics.jsonl`576–611. The old partial's uncheckpointed576–586 are **11 discarded records**, replaced rather than concatenated.

J_G uses only finalized origin `train_metrics.jsonl` files,611 rows each. Earlier CPU J_G306 histories are separate experiments; the old collected GPU J_G14 log ending428 is excluded. Final resume generations5/6/3 are provenance, not independent training counts. Fourteen GPU discarded tails are described with source/receipt hashes and exact ranges/counts in the manifest. J_G15 workers32 generation5 rows151–154 were explicitly discarded; generation6 resumed checkpoint150 with16workers. The optional historical full pre-revert copy is identified by hash and a logical relative alias that removes machine identifiers.

All resumes restore algorithm state only, **not bitwise environment/cache/episode/RNG continuity**. CPU segment RNG resets and GPU incomplete seed-audit traces remain limitations; complete row coverage is not evidence of uninterrupted training or restored original protocol eligibility. Historical J_G `eligible_for_main_completeness=false` is not overturned by this export.

## Offline rebuild

From any working directory, with Python, NumPy and Matplotlib already available:

```sh
python /path/to/review/analysis/training_progress.py --output-dir /path/to/output
python /path/to/review/analysis/training_progress.py --validate-only
python -m unittest discover -s /path/to/review/tests -p test_training_progress.py -v
```

Without `--output-dir`, output goes to the script-relative `review/build/training`. Inputs default to script-relative `review/data/training`; `--data-dir` can point to another complete copy of those shipped inputs. The analysis reads only those two files, imports no project science/model code, and does not need inventory, local sources, author exporter, weights, simulator or Ray.

Reference outputs are `results/training/{pv_speed_mean,team_return_mean}.{pdf,svg,png}`. Each has five family panels and three seed lines, common scales, seed legend, selected-checkpoint diamonds, and resume ticks/circles. Every accepted segment is a separate curve artist: there is no line across a resume boundary and no zero-filled gap. These are the raw saved rolling metrics, without further smoothing or confidence bands. Accepted steps exclude rejected computation; they are not total compute cost, continuous wall time, independent observations, or a causal budget comparison. Selection markers identify receipt-selected checkpoints, not maxima chosen from these curves.

Verified here: Python3.11.10, NumPy2.1.2, Matplotlib3.10.0, DejaVu Sans. No package installation was performed. PDF/SVG dates are removed and SVG IDs use a fixed salt. Figure bytes matched across two isolated same-runtime rebuilds; cross-version font/rendering or gzip-library byte equality is not promised. Eight tests cover coverage/counters, accepted ranges/exclusions, markers, source/hash schema, negative cases, corruption, float/gzip replay, separate plot segments, and an isolated CLI build. Third-party Pyparsing deprecation warnings did not affect results.

`provenance/training_export.json` records the original inventory hash, input/output sizes, verified versions, exporter hash and metric-definition source hashes. `provenance/training_verification.json` records checks and reference hashes. `provenance/training_commands.md` records execution commands. The author-only `_local/export_training.py` streams one original JSONL record at a time, extracts only allowed fields, verifies original hashes against inventory and rechecks each primary source after parsing. It is not needed for public rebuilds. Compact evidence replaces entire operator logs/contracts; no license terms are invented here.
