# Selected policy weights

The public weights contract contains **21 selected configurations and all 31 numeric policy states** in [selected_policies.zip](<weights/selected_policies.zip>), indexed by [MODELS.json](<weights/MODELS.json>). The NPZ members preserve the original accepted-MC weights byte for byte; they are not converted checkpoints. No optimizer state or pickle is included.

From the reproduction directory, verify the package with Python's standard library:

```sh
python3 scripts/verify_weights.py weights
```

The verifier reads only the supplied weights directory. It checks the fixed selection matrix, role ownership, parameter totals, archive/member sizes and SHA-256 hashes, exact ZIP member sets, and NPY names, float32 dtypes, shapes and payload lengths. It rejects duplicate JSON/ZIP keys, unsafe paths and non-file members. It does not extract files, load a framework model, or inspect private sources. Hashes establish consistency with the supplied index, not independent authenticity of that index. Array keys and shapes are checked against the index derived from the original manifests.

| Family | Seeds → selected iterations | Included roles |
|---|---|---|
| A | 20260914→500, 20260915→600, 20260916→450; 2026102804→550, 2026102805→450 | AV |
| P_L | 20260914→550, 20260915→500, 20260916→500 | PV |
| P_G | 20260914→500, 20260915→450, 20260916→450 | PV |
| J_L | 20260914→450, 20260915→550, 20260916→450 | AV + PV |
| J_G | 20260914→450, 20260915→500, 20260916→450; 2026102804→450, 2026102805→450 | AV + PV |
| J_NoMP | 2026102804→450, 2026102805→500 | AV + PV |

Each AV state has **355,460** parameters; local PV states have **23,812**, graph PV states **411,012**, and NoMP PV states **411,572**. `D2_L` reuses the selected `J_L` AV state for the same seed with native PV behavior; it has no additional archive member or learned PV state.

For an optional numeric dictionary reader, NumPy must already be available in the recipient's environment:

```python
from policies import load_state_dict, verify_weights

index = verify_weights('weights')          # standard library only
state = load_state_dict('weights', 'J_L', 20260914, 'av_policy')
# state maps original parameter names to NumPy arrays.
# Equivalent reused AV state:
d2_state = load_state_dict('weights', 'D2_L', 20260914, 'av_policy')
```

The reader validates the whole package before importing NumPy, then reads the validated NPZ bytes using `allow_pickle=False`. Missing/native roles raise `ValueError`. A numeric dictionary is not an executable policy: a forward pass requires the matching model class, architecture and observation encoding. Use the matching source from the **same repository commit**: [local/graph model definitions](<../code/sep_research/models.py>), [CPU inference helpers](<../code/sep_research/inference.py>) and [the exact NoMP network](<../code/sep_research/nomp_model.py>). The generic `build_pv_model` helper supports local/graph encoders, not NoMP: use `PvNoMPNetwork` for `relation_set_mlp`, never substitute J_G. The model-source hashes are recorded in the index. Source references here concern the full Git checkout; a standalone saved-data export does not include the full simulator or model implementation tree.

Original snapshot provenance flags (including `incomplete` or `budget_reached=false` where present) are retained, not promoted: selected checkpoints can precede completion of the full training lineage. The `env` fields are selected public constants, not a complete environment/optimizer restore contract.

This package performs no inference, simulation, MC evaluation or training. The reader does not establish training/resume reproducibility or change original provenance flags. Scientific source comparisons and real-array checks are separate release verification work.

The synthetic verifier regression tests need only the standard library:

```sh
python3 -B -m unittest discover -s tests -p 'test_weights*.py' -v
```
