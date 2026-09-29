# Full-package integration verification

**PASS.** The parent verification used a new manifest-only temporary export, not the working research tree. Every top-level test/rebuild command ran with `python -I` from an unrelated directory under a Python audit hook denying file/directory access beneath the original workspace. A deliberate source-read probe was denied. This is exercised Python-level isolation, not OS mount isolation or a claim about every possible native-library syscall.

## Tests and reconstruction

| Check | Result |
|---|---:|
| Safe release/export regression tests | 14 PASS |
| Publication statistics/input tests | 13 PASS |
| Training-history/rebuild tests | 8 PASS |
| Bounded scientific acceptance | 9 PASS |
| Scenario seed/grid manifest | 1 PASS |
| **Total** | **45 PASS, no skips** |

The scientific acceptance used synthetic weights, 15 counted phase/transition calls, three real compiled-kernel calls and 44 verified project-module imports. A separate component check regenerated one low-density initial case with 1,000 warmup steps and matched the historical physical and role hashes. No training, checkpoint loading, full scenario-grid generation or new measurement campaign was run.

Both offline CLIs were then executed from the clean export:

```sh
python -I analysis/replay_publication.py
python -I analysis/training_progress.py
python -I scripts/release.py verify
```

Twenty-one regenerated reference files matched byte-for-byte: four tables, five publication PNGs, six public data summaries, and six training figures (two metrics × PDF/SVG/PNG). An additional direct comparison against the untouched manuscript matched all fourteen applicable originals: four tables, five PNGs and five statistical JSON files. The public selection receipt intentionally removes original checkpoint paths; it is not claimed byte-identical to the private selection file. Scientific payloads retain their immutable reference checks. Publication vector figures were regenerated; their changing metadata is not a byte-equality claim.

The two training figures were visually inspected: all five panels and three seed lines are readable, with separate resume segments and selected-checkpoint markers. Original manuscript figures remain unchanged.

## Preservation and publication boundary

All **471 protected original files** in the parent baseline retained their SHA256 hashes after integration. This baseline covers tracked originals except the two explicitly allowed navigation/ignore edits, plus manuscript outputs/scripts/PDFs/audits. Scientific inventories and exporters additionally checked their original raw/history/source hashes. This is not a claim to hash every byte of the approximately51GB local workspace.

The root changes are limited to `.gitignore` and a README navigation link. The local experimental tree, checkpoints, raw logs and previous Git history remain in place. Generated caches were excluded from release manifests, not deleted. The Git candidate set matched the explicit release file list plus `MANIFEST.json`; local inventories, author tools and build outputs remained ignored.

Final assembly pins the verified analysis dependencies and explicitly redacts the private host path in the historical science verification command. Original numerical scientific copies remain byte-identical. Some historical contract-gate source retains non-secret host/registry labels; the science guide explains their scope. The release content scan checks obvious key/token patterns, not all conceivable secrets, and the old Git history has not received an exhaustive secret audit.

Two packaging integration defects were found and corrected: the compressed scientific input needed a separate bounded decompression limit (50MiB, while keeping10MiB per published file), and the training format required `.jsonl.gz` support. Both are covered by regression tests. These changes affect publication packaging, not scientific computations.

After this receipt and final explanatory documentation were written, the unchanged tested code/data were resealed and copied to the final package-only export, whose manifest is verified separately. License selection, approval of public metadata and a permanent release URL/DOI remain author decisions. No commit, push, Git-history rewrite or deletion of original research artifacts was performed.
