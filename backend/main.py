"""
main.py — AQI Forecasting & Explainable AI System (Noida)
===========================================================
Central CLI orchestrator for the entire pipeline.

Usage examples
--------------
    python main.py --stage preprocess --config configs/default.yaml
    python main.py --stage all       --config configs/default.yaml
    python main.py --stage preprocess --dry-run
    python main.py --stage preprocess --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

VALID_STAGES = ["preprocess", "features", "train", "evaluate", "explain", "dashboard", "all"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aqi-forecast",
        description="AQI Forecasting & XAI System — Noida",
    )
    parser.add_argument("--stage", type=str, default="all",
                        choices=VALID_STAGES, help="Pipeline stage to execute.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--station", type=str, default=None,
                        help="Override station_id from config.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_preprocess(config, station_id=None) -> None:
    from src.preprocessing.pipeline import run_preprocessing_pipeline
    run_preprocessing_pipeline(config, station_id=station_id, save_outputs=True)


STAGE_REGISTRY: Dict[str, Callable] = {
    "preprocess": _run_preprocess,
    # "features":  _run_features,   # Phase 3
    # "train":     _run_train,       # Phase 5
    # "evaluate":  _run_evaluate,    # Phase 5
    # "explain":   _run_explain,     # Phase 7
    # "dashboard": _run_dashboard,   # Phase 8
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Config + logging bootstrap
    from src.utils.config import load_config
    from src.utils.logger import configure_logging

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    configure_logging(
        log_dir=Path(cfg.paths.logs),
        station_id=cfg.project.station_id,
        level="DEBUG" if args.verbose else cfg.project.log_level,
    )

    if args.dry_run:
        print(f"[DRY RUN] Would execute stage(s): "
              f"{'all' if args.stage == 'all' else args.stage}")
        sys.exit(0)

    stages = list(STAGE_REGISTRY.keys()) if args.stage == "all" else [args.stage]
    sid = args.station  # None → each stage uses config default

    for stage in stages:
        if stage not in STAGE_REGISTRY:
            print(f"[WARN] Stage '{stage}' not yet implemented — skipping.")
            continue
        STAGE_REGISTRY[stage](cfg, station_id=sid)

    sys.exit(0)


if __name__ == "__main__":
    main()
