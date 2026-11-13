"""CPU-only comparison of uninterrupted and resumed complete training states."""

import argparse
import json
from pathlib import Path

import torch


def compare(a, b, path="", differences=None, counts=None):
    differences = [] if differences is None else differences
    counts = {"tensors": 0, "tensor_bytes": 0} if counts is None else counts
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        counts["tensors"] += 1
        counts["tensor_bytes"] += a.numel() * a.element_size()
        if a.shape != b.shape or a.dtype != b.dtype:
            differences.append({"path": path, "kind": "tensor schema"})
        elif not torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8),
            b.contiguous().reshape(-1).view(torch.uint8),
        ):
            finite = torch.isfinite(a) & torch.isfinite(b)
            delta = (a[finite].double() - b[finite].double()).abs()
            differences.append(
                {
                    "path": path,
                    "kind": "tensor bytes",
                    "max_abs_finite": float(delta.max()) if delta.numel() else None,
                }
            )
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            differences.append({"path": path, "kind": "keys"})
        for k in a.keys() & b.keys():
            compare(a[k], b[k], path + "/" + str(k), differences, counts)
    elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            differences.append({"path": path, "kind": "length"})
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + "/" + str(i), differences, counts)
    elif a != b:
        differences.append({"path": path, "kind": "value"})
    return differences, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("uninterrupted", type=Path)
    p.add_argument("resumed", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    a, b = [
        torch.load(x, map_location="cpu", weights_only=True)
        for x in (args.uninterrupted, args.resumed)
    ]
    # Guard evidence records the distinct process histories and is not resumable state.
    a.pop("guard_evidence")
    b.pop("guard_evidence")
    diffs, counts = compare(a, b)
    result = {
        "uninterrupted": str(args.uninterrupted),
        "resumed": str(args.resumed),
        "excluded": "guard_evidence only (process-local attestations)",
        "compared": counts,
        "bitwise_equal": not diffs,
        "differences": diffs,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
    raise SystemExit(bool(diffs))


if __name__ == "__main__":
    main()
