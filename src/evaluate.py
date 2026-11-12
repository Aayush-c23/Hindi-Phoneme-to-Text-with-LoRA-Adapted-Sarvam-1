"""Evaluate phoneme-to-text predictions: exact match, corpus CER/WER.

Accepts prediction JSONL rows with either:
  {"prediction": ..., "target": ...}   (recomputes everything), or
  {"exact": bool, "cer": float, "wer": float}  (aggregates stored per-row rates)
"""

import argparse
import json
import re


def _norm(text: str) -> str:
    return re.sub(r"[^\u0900-\u097F\s]", "", text or "").strip()


def _lev(a, b) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[n]


def evaluate_file(path: str) -> dict:
    n = 0
    exact = 0
    char_edits = 0
    ref_chars = 0
    word_edits = 0
    ref_words = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            n += 1
            if "prediction" in o and "target" in o:
                pred, ref = _norm(o["prediction"]), _norm(o["target"])
                if pred == ref:
                    exact += 1
                char_edits += _lev(list(pred), list(ref))
                ref_chars += len(list(ref))
                word_edits += _lev(pred.split(), ref.split())
                ref_words += len(ref.split())
            else:
                if o.get("exact"):
                    exact += 1
                char_edits += float(o.get("cer", 0.0))
                ref_chars += 1
                word_edits += float(o.get("wer", 0.0))
                ref_words += 1
    return {
        "n": n,
        "exact_pct": 100.0 * exact / n if n else 0.0,
        "cer_pct": 100.0 * char_edits / ref_chars if ref_chars else 0.0,
        "wer_pct": 100.0 * word_edits / ref_words if ref_words else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Score prediction JSONL")
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()
    m = evaluate_file(args.predictions)
    print(f"n={m['n']} exact={m['exact_pct']:.2f}% CER={m['cer_pct']:.2f}% WER={m['wer_pct']:.2f}%")


if __name__ == "__main__":
    main()
