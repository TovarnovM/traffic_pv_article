# Publication boundary and remaining release decisions

## This package is separate from the working research tree

The local project retains its original pipeline, experiments, logs, checkpoints and manuscript. The review package contains selected copies and compact exports, not a relocation of those originals. Source hashes and any portability-only adaptations must be documented in `provenance/`.

There are distinct reproduction levels:

1. **Replay published measurements:** recompute statistics, tables and figures from the shipped per-road results, without trained models or a simulator campaign.
2. **Check implementation:** run representative CPU tests of the included scientific code. This is limited acceptance, not exhaustive validation of every branch or historical runtime.
3. **Repeat experiments:** use the documented initial conditions, code and training/evaluation recipes. This requires substantial computation and, for evaluation of the exact selected policies, their weights. Packaging tests do not constitute a new training or evaluation campaign.

A file that records a hash or seed is not automatically a saved initial state or a restorable checkpoint. The data documentation must distinguish these explicitly. Training histories are observations of the accepted runs; resumed and missing segments must not be silently converted into a continuous from-scratch trajectory.

## No automatic publication of the existing Git history

The existing branch contains hundreds of previously tracked files, including approximately 121 MB of older artifacts. Those files and earlier commits would remain part of a normal descendant branch even after adding `.gitignore` rules. The local inventory is not an exhaustive secret audit of that history.

For a package-only release, use a verified export of the explicit review manifest into an empty directory **without the parent `.git` directory**. A fresh public repository can subsequently be initialized there after author approval. This avoids rewriting or deleting the working repository's history. Preparing an export does not authorize a commit or push.

Local inventories, source-export utilities, caches and test/build scratch files are not release inputs. The final allowlist and integrity check, rather than an archive of the whole workspace, define the publication boundary.

## Licensing is an author decision still pending

No project license was established by the repository inventory. Preparation of files does not select or grant an open-source license on behalf of the authors. Before public release, confirm rights to the included project code/data and select suitable terms. Do not apply those terms to third-party code or assets without checking their existing licenses.

The package must not include third-party journal PDFs, local environment secrets, infrastructure/container inspection dumps, transport credentials, or unrelated archives. Dependencies should be installed separately rather than vendored indiscriminately. A runtime version list is not a third-party redistribution audit.

## Stable citation

Once approved and published, create a versioned release and cite its permanent URL (preferably an archived DOI as well). Do not invent a repository URL, DOI, license or release date before those decisions are made. A moving branch alone is not a stable reference for the article.
