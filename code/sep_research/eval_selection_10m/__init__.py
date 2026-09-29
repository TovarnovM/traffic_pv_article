"""Labelled 10M CPU-matrix checkpoint-selection variant (12 pairs; J_G excluded).

This package is a declared variant of sep_research.admission's 20M 'main'
selection path. It never reads, writes, shells out to or relaxes the frozen 20M
anchor (sep_research/manifests/main_profile.json, FINAL_ITERATION=1221, all 15
pairs). It mirrors that path's semantics for the CPU 10M campaign:

* families A, P_L, P_G, J_L x training seeds 20260914/15/16 = 12 pairs;
* candidates contract.candidates(611) = 450, 500, 550, 600, 611;
* validation split only (frozen 20-run manifest, root 610003, rho 0.30, AV 1.00);
* selection rule PV_GRAPH_POST_TRAINING_PLAN.md 4.5.

J_G is deliberately excluded until its three lines reach iteration 611.
"""
