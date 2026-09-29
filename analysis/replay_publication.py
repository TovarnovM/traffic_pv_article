"""Rebuild the publication from shipped scalar observations and winner receipts."""
import argparse
import json
from pathlib import Path
import sys

# Explicit local imports also work under python -I, from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import publication as main
import publication_inputs as inputs


def verify_payloads(payloads, reference):
    for name, value in payloads.items():
        if main.canonical(value) != reference['metric_sha256'][name]:
            raise ValueError(f'immutable publication metric mismatch: {name}')


def replay(output_dir):
    indexed, receipt, metadata, reference = inputs.load_inputs()
    conditions, contrasts, background, laps = main.calculate(indexed)
    summary = main.summarize(conditions, contrasts, background, laps)
    payloads = dict(conditions=conditions, contrasts=contrasts, background=background,
                    laps=laps, summary=summary, selection=receipt['winners'])
    verify_payloads(payloads, reference)
    for name in ('conditions', 'contrasts', 'background', 'laps'):
        main.write_json(output_dir / 'data/main_series' / f'{name}.json',
                        dict(metadata=metadata, records=payloads[name]))
    main.write_json(output_dir / 'data/main_series/summary.json', dict(metadata=metadata, **summary))
    main.write_json(output_dir / 'data/main_series/selection.json', receipt)
    main.tables(summary, receipt['winners'], output_dir)
    for name, digest in reference['table_sha256'].items():
        if main.sha(output_dir / name) != digest:
            raise ValueError(f'immutable publication table mismatch: {name}')
    figures = main.plot_figures(conditions, contrasts, background, laps, output_dir)
    main.write_json(output_dir / 'replay_checks.json', dict(
        logical_rows=len(indexed), physical_rows=29700, selected_models=15,
        metric_hashes_match=True, four_tables_byte_identical=True, figures=figures,
        bootstrap_indices_sha256=main.hashlib.sha256(main.indices().tobytes()).hexdigest(),
        module_files={m.__name__: str(Path(m.__file__).relative_to(inputs.REVIEW))
                      for m in (main, inputs, sys.modules['publication_style'])}))
    print(json.dumps(dict(output_dir=str(output_dir), logical_rows=len(indexed),
                          metric_hashes_match=True, four_tables_byte_identical=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=inputs.REVIEW / 'build/publication')
    args = parser.parse_args()
    replay(args.output_dir.resolve())
