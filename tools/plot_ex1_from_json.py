#!/usr/bin/env python3
"""CLI: generate EX1 plots from a JSON file produced by statistics_exp_1

Usage:
    python tools/plot_ex1_from_json.py path/to/EX1_dataset_L_loss_data.json

If you don't provide an output directory, plots are written to results/ex_1.
"""
import argparse
import os
import json
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import Metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jsonfile', help='Path to EX1 JSON data file')
    p.add_argument('--out', help='Output directory (default: results/ex_1)', default=None)
    p.add_argument('--ymin', type=float, default=50.0)
    p.add_argument('--ymax', type=float, default=100.0)
    p.add_argument('--ystep', type=float, default=5.0)
    args = p.parse_args()

    if not os.path.exists(args.jsonfile):
        raise SystemExit(f"JSON file not found: {args.jsonfile}")

    with open(args.jsonfile, 'r') as f:
        data = json.load(f)

    Metrics.plot_ex1_from_data(data, out_dir=args.out, y_min=args.ymin, y_max=args.ymax, y_step=args.ystep)


if __name__ == '__main__':
    main()
