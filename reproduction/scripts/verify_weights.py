"""Verify the public selected-policy weights using only the Python standard library."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policies import verify_weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights_dir', type=Path)
    args = parser.parse_args(argv)
    try:
        verify_weights(args.weights_dir)
    except ValueError as exc:
        print(f'weights: {exc}', file=sys.stderr)
        return 1
    print('weights: verified 21 models / 31 policy states')
    return 0


if __name__ == '__main__':
    sys.exit(main())
