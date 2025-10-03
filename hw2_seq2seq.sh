#!/usr/bin/env bash
# Usage: ./hw2_seq2seq.sh <test_feat_dir> <output_txt>
set -e
python3 test_run.py "$1" "$2"
