# How to Run `hw2_seq2seq.sh`

This script runs **inference + BLEU evaluation** on the MSVD test set using the trained model.

## Prerequisites

- Python 3.9+ and PyTorch installed
- Repo files present:
  - `models.py`, `test_run.py`, `bleu_eval.py`
  - `hw2_seq2seq.sh` (this script)
- Dataset files placed next to the repo root:
  - `testing_data/feat/*.npy`
  - `testing_label.json`
- A **trained model** and vocab mapping created by training:
  - `SavedModel/model0.h5`
  - `SavedModel/i2wData.pickle`

> I tarined it, with:
> ```bash
> python3 main.py --epochs 100 --batch 5 --num_workers 0
> ```

## Basic Usage

From the project root:

```bash
chmod +x hw2_seq2seq.sh
./hw2_seq2seq.sh testing_data/feat test_output.txt
