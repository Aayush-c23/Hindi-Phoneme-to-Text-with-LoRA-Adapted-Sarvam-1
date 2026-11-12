"""Baseline: evaluate base sarvamai/sarvam-1 (no adapter) on the 4 splits.

Registers the four prompt tags on the base tokenizer (new random embeddings)
and generates with beam search.

Usage:
    python src/evaluate_base.py --data_dir /path/to/jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from data_utils import TAGS
except ImportError:
    from src.data_utils import TAGS

BASE_MODEL_ID = "sarvamai/sarvam-1"
SPLITS = ["val_clean.jsonl", "val.jsonl", "test_clean.jsonl", "test.jsonl"]


def extract_phonemes(prompt: str) -> str:
    return prompt.split("<PHONEMES>\n", 1)[1].split("\n</PHONEMES>", 1)[0].strip()


def build_prompt(phonemes: str) -> str:
    return f"{TAGS[0]}\n{TAGS[1]}\n{phonemes}\n{TAGS[2]}\n{TAGS[3]}\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate base model, no adapter")
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    missing = [s for s in SPLITS if not (data_dir / s).exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files in {data_dir}: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": TAGS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16 = bool(torch.cuda.is_available()
                and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if bf16 else (
            torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    device = next(model.parameters()).device if torch.cuda.is_available() \
        else torch.device("cpu")
    print("device:", device, flush=True)

    for split in SPLITS:
        rows = [json.loads(l) for l in open(data_dir / split, encoding="utf-8") if l.strip()]
        print(f"\n=== {split} (base, no adapter) ===\nrows: {len(rows)}", flush=True)
        out_path = data_dir / f"{Path(split).stem}_base_predictions.jsonl"
        with open(out_path, "w", encoding="utf-8") as out:
            for i, row in enumerate(rows, 1):
                phonemes = extract_phonemes(row["prompt"])
                target = " ".join(row["target"].split())
                inputs = tokenizer(build_prompt(phonemes), return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=64, do_sample=False, num_beams=3,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id)
                prediction = " ".join(tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True).split("\n")[0].split())
                out.write(json.dumps({"phonemes": phonemes, "target": target,
                                      "prediction": prediction,
                                      "exact": prediction == target},
                                     ensure_ascii=False) + "\n")
                if i % 25 == 0:
                    print(f"  processed {i}/{len(rows)}", flush=True)
        print(f"wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
