import argparse
from pathlib import Path

from cvd70k.evaluation.report import generate_reports
from cvd70k.visualization.figures import render_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    generate_reports(Path.cwd(), not args.skip_bootstrap)
    if not args.skip_figures:
        render_all(Path.cwd())
    print("Reports and requested figures generated", flush=True)
