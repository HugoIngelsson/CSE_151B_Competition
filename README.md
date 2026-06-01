# CSE 151B Competition — Setup Guide

The goal is to run inference with **Qwen3-4B-Thinking-2507** on mathematics questions.

## Contents

| File | Description |
|---|---|
| `run_inference.py` | Main entry point |
| `judger.py` | Needed for validation and parsing |
| `utils.py` | Utilities used by `judger.py` |
| `data/private.jsonl` | Dataset with `question`, `id`, and possibly `options` as fields|
| `result.csv` | Output file containing final answers |

## Setup

Note that to run this, a CUDA compatible GPU is needed. 

Optionally, you may choose to start a virtual environment. However, our team didn't do this on DSMLP as it led to disk quota exceeded errors. 

Install the required packages in `requirements.txt`.

This is possible with `uv` or `pip`. For example, run

```bash
pip install -r requirements.txt
```

To start inference, run
```bash
python run_inference.py
``` 

If you wish to see more debug statements, run 

```bash 
python run_inference.py debug
```

## Runtime and Rerunning
When using an **NVIDIA l40s** on [DSMLP](https://datahub.ucsd.edu/hub/login?next=%2Fhub%2F), inference took ~20 hours to run on the full private dataset of **943** questions. 

Note that `run_inference.py` generates responses in batches of 100, so if execution is terminated, intermediate results still may be saved. 

If execution is terminated, rerunning `run_inference.py` will continue to run inference **only on unanswered questions**. Because of this, it is important **not to delete/modify `result.csv`**, or else progress may be lost.