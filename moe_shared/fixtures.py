"""Durable fixture report; explicitly not real-pretrained acceptance."""
import argparse
import json
from pathlib import Path
import time
import unittest

from .audit import resource_snapshot


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    before = resource_snapshot()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests"))
    report = {"model_kind": "tiny random fixture; NOT pretrained-model acceptance",
              "tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "passed": result.wasSuccessful() and not result.skipped,
              "wall_seconds": time.perf_counter()-start,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "resources_before": before, "resources_after": resource_snapshot()}
    (output / "fixture-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
