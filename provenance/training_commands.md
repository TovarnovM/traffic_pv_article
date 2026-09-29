# Training export and verification commands

Executed in the existing container via the following host prefix (no container lifecycle operations). `$WORKSPACE` denotes the local repository root and is not a public machine-path identifier:

```sh
WORKSPACE=/path/to/traffic_rl
devcontainer exec --workspace-folder "$WORKSPACE" \
  --config "$WORKSPACE/sep_research/gpu_duplicate_20260916/runtime/devcontainer.json" \
  env CUDA_VISIBLE_DEVICES= MPLCONFIGDIR=/tmp/traffic_rl_review_mpl \
  NUMBA_CACHE_DIR=/tmp/traffic_rl_review_numba PYTHONDONTWRITEBYTECODE=1 \
  python review/_local/export_training.py
```

With the identical prefix through `python`, subsequent commands were:

```sh
review/analysis/training_progress.py --output-dir review/results/training
-m unittest discover -s review/tests -p test_training_progress.py -v
```

The final verification used a Python standard-library driver in the same container: exported twice and compared hashes; copied only `analysis/training_progress.py`, `tests/test_training_progress.py`, and `data/training/` into a fresh temporary review directory; ran its unittest suite from an unrelated temporary working directory; rebuilt its six figures; compared every figure byte hash with `results/training/`. The test suite additionally builds twice in an isolated temporary layout and compares all six output hashes. No temporary copy contains the local exporter or original science inputs. Original inputs were opened only for the author export and read-only inventory/schema inspection, never by public rebuilds.

Public provenance uses relative logical paths, hashes and small allowlisted fields; home paths, machine/node identifiers and entire operator logs/contracts are not exported. The exceptional historical pre-revert log path is replaced with `historical/J_G-20260915/workers32_before_revert.jsonl`; its original SHA-256 preserves identity. All originals remain in place. No deletion, move, commit, push, SSH, install, model loading, training or simulation was performed.
