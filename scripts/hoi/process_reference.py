#!/usr/bin/env python3
"""Generate the Part B minimal reference in a new, isolated output directory."""
import argparse
from pathlib import Path

from supermimic.hoi.reference import (
    DEFAULT_ASSET, DEFAULT_HOI, DEFAULT_OUTPUT, DEFAULT_ROBOT, convert,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', type=Path, default=DEFAULT_ROBOT)
    parser.add_argument('--hoi', type=Path, default=DEFAULT_HOI)
    parser.add_argument('--asset-manifest', type=Path, default=DEFAULT_ASSET)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = convert(args.robot, args.hoi, args.asset_manifest, args.output_dir)
    print(f'B1 PASS: {args.output_dir / "reference.npz"}\nSHA256 {result["sha256"]}')


if __name__ == '__main__':
    main()
