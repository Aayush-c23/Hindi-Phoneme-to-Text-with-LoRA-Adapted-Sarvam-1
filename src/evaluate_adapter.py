"""Evaluate a trained LoRA adapter on clean/noisy phoneme-to-text splits.

Loads the adapter via HindiPhonemeTranslator, generates Devanagari for each
prompt's phonemes, and writes per-split prediction JSONL plus corpus
exact/CER/WER.

Usage:
    python src/evaluate_adapter.py --adapter_dir /path/to/sarvam_lora --data_dir /path/to/jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))
from punarvaani_sarvam import HindiPhonemeTranslator  # noqa: E402

from evaluate import evaluate_file  # noqa: E402

SPLITS = ["val_clean.jsonl", "val.jsonl", "test_clean.jsonl", "test.jsonl"]


def extract_phonemes(prompt: str) -> str:
    return prompt.split("<PHONEMES>\n", 1)[1].split("\n</PHONEMES>", 1)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter on 4 splits")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--data_dir", required=True,
                        help="Folder holding val_clean/val/test_clean/test JSONL")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    missing = [s for s in SPLITS if not (data_dir / s).exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files in {data_dir}: {missing}")

    translator = HindiPhonemeTranslator(adapter_path=args.adapter_dir,
                                        device="cuda")
    if translator.is_mock:
        raise RuntimeError(f"Adapter did not load: {translator.mock_reason}")

    adapter_name = Path(args.adapter_dir).name
    for split in SPLITS:
        rows = [json.loads(l) for l in open(data_dir / split, encoding="utf-8") if l.strip()]
        print(f"\n=== {split} ===\nrows: {len(rows)}", flush=True)
        out_path = data_dir / f"{Path(split).stem}_{adapter_name}_predictions.jsonl"
        with open(out_path, "w", encoding="utf-8") as out:
            for i, row in enumerate(rows, 1):
                phonemes = extract_phonemes(row["prompt"])
                target = " ".join(row["target"].split())
                prediction = " ".join(translator.translate(
                    phonemes, max_new_tokens=args.max_new_tokens).split())
                out.write(json.dumps({"phonemes": phonemes, "target": target,
                                      "prediction": prediction,
                                      "exact": prediction == target},
                                     ensure_ascii=False) + "\n")
                if i % 25 == 0:
                    print(f"  processed {i}/{len(rows)}", flush=True)
        m = evaluate_file(str(out_path))
        print(f"exact={m['exact_pct']:.2f}% CER={m['cer_pct']:.2f}% "
              f"WER={m['wer_pct']:.2f}%\nwrote: {out_path}")


if __name__ == "__main__":
    main()
