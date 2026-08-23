"""Run every model in one dataset cluster back-to-back and compare.

    python benchmarks/run_cluster.py --cluster cifar --device auto
    python benchmarks/run_cluster.py --cluster mnist --device auto
    python benchmarks/run_cluster.py --cluster segmentation --device auto
    python benchmarks/run_cluster.py --cluster detection --device auto

Each models/<name>/example.py must print one final line:

    RESULT: model=<name> metric_name=<name> metric=<value> params=<n> train_time_s=<value>

which this script parses to build the comparison table.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULT_RE = re.compile(
    r"RESULT: model=(?P<model>\S+) metric_name=(?P<metric_name>\S+) "
    r"metric=(?P<metric>\S+) params=(?P<params>\S+) train_time_s=(?P<train_time_s>\S+)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", required=True, choices=["cifar", "mnist", "segmentation", "detection"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "benchmarks" / "configs" / f"{args.cluster}_suite.yaml").read_text())
    epochs = args.epochs or config.get("epochs", 30)

    rows = []
    for name in config["models"]:
        script = ROOT / "models" / name / "example.py"
        print(f"\n=== {name} ===")
        proc = subprocess.run(
            [sys.executable, str(script), "--device", args.device, "--epochs", str(epochs)],
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            continue
        match = RESULT_RE.search(proc.stdout)
        if match:
            rows.append(match.groupdict())
        else:
            print(f"[warn] no RESULT: line found in {name}'s output", file=sys.stderr)

    if not rows:
        print("No results collected.")
        return

    print(f"\n{'model':>14} | {'metric_name':>12} | {'metric':>10} | {'params':>10} | {'train_time_s':>12}")
    for r in rows:
        print(
            f"{r['model']:>14} | {r['metric_name']:>12} | {r['metric']:>10} | "
            f"{r['params']:>10} | {r['train_time_s']:>12}"
        )


if __name__ == "__main__":
    main()
