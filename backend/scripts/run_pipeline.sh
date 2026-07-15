#!/bin/bash
# scripts/run_pipeline.sh — End-to-end pipeline runner (CI/CD friendly)
set -e
python main.py --stage all --config configs/default.yaml
