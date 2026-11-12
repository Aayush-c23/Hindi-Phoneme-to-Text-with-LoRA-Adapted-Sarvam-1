# Noise-Robust LoRA Adaptation of Sarvam-1 2B for Hindi Phoneme-to-Text

Fine-tuned [`sarvamai/sarvam-1`](https://huggingface.co/sarvamai/sarvam-1) (2B) with LoRA to
reconstruct fluent Devanagari Hindi from phoneme sequences — including noisy phoneme
sequences that simulate visual-speech (lip-reading) errors.

## What this is

A phoneme is a speech sound (`N`, `AA`, `TD`). On the lips many Hindi sounds are visually
identical (`P`/`B`/`M` are all bilabial closures — one viseme). A phoneme-to-text model must
therefore correct systematic substitutions, drops, and duplications using Hindi vocabulary
and grammar. This project trains exactly that mapping:

```text
<S2S>
<PHONEMES>
N A M A S TD E | BH AA R A TD
</PHONEMES>
<TEXT>
नमस्ते भारत
```

## Results

Adapter: best checkpoint at **step 15000** (val loss rose at 15.5k, ran to 16k to confirm
overfitting, kept 15k). LoRA r=16, alpha=32, bf16 on A5000 24 GB.

| Split | Exact match | CER | WER |
|---|---:|---:|---:|
| Val clean (4,000) | 58.10% | 1.49% | 4.76% |
| Test clean (2,000) | 57.70% | 1.42% | 4.71% |
| Val noisy (4,000) | 33.20% | 3.94% | 9.95% |
| Test noisy (2,000) | 31.00% | 3.91% | 9.94% |

Metrics are corpus-level (total edits / total reference length); exact match is strict
whole-sentence equality. Val and test agree closely — the adapter generalizes, not just fits val.

Base Sarvam-1 without the adapter scores 0.00% exact / ~200% CER/WER: the four prompt tags
(`<S2S>`, `<PHONEMES>`, `</PHONEMES>`, `<TEXT>`) are new tokens with random embeddings
zero-shot, so the task is not solvable without fine-tuning.

**Scope note:** noisy prompts use *synthetic* viseme noise (12% swap / 3% drop / 2% dup),
a proxy for lip-reading errors — not real ASR output. Clean prompts use exact G2P phonemes.

## Data

Curated from 200,000 Hindi source sentences → 104,299 high-quality → 66,000 final rows:

| Split | Rows | Purpose |
|---|---:|---|
| Train | 60,000 | 80% noisy / 20% clean prompts |
| Val | 4,000 | noisy prompts, checkpoint selection |
| Test | 2,000 | noisy prompts, final test (plus 2k clean counterparts per split) |

- Filters: 6–22 words, 30–150 chars, pure Devanagari, no rare chars, no repeated matras,
  unique-word ratio ≥ 0.6, valid G2P.
- **Zero target overlap** between train/val/test (verified by exact target match).
- Train noise sampled per row: swap 8–18%, drop 2–5%, dup 1–3%; eval noise fixed at
  12%/3%/2%. Swaps stay inside viseme groups (`P/B/M`, dentals, velars, ...).
- See `data_samples/` for format examples (30 train + 10 val rows) and
  `results/demo_predictions10.jsonl` for 10 prediction examples.

## Repo layout

```text
sarvam-lora-hindi/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── train_lora.py        # LoRA training (PEFT + HF Trainer)
│   ├── evaluate.py          # exact / CER / WER over prediction JSONL
│   ├── evaluate_adapter.py  # run trained adapter over the 4 splits
│   ├── evaluate_base.py     # base-model baseline over the 4 splits
│   └── data_utils.py        # prompt contract + viseme-noise helpers
├── vendor/
│   └── punarvaani_sarvam/
│       └── __init__.py      # vendored Sarvam.py (translator + trainer, pinned)
├── scripts/
│   └── make_samples.py      # rebuild data_samples/ from full JSONL (private)
├── data_samples/
│   ├── train_sample30.jsonl
│   └── val_sample10.jsonl
└── results/
    ├── metrics.json         # the table above, machine-readable
    └── demo_predictions10.jsonl
```

Adapter weights (~1.6 GB with saved tag embeddings) are **not** in this repo (GitHub file
limits + weight-hosting policy). The adapter config (`r=16`, target modules, saved
`embed_tokens`/`lm_head`) is documented in `results/metrics.json`.

## Reproduce

```bash
pip install -r requirements.txt

# train (needs sarvamai/sarvam-1 access + CUDA GPU)
python src/train_lora.py \
  --train_jsonl /path/to/train.jsonl \
  --val_jsonl /path/to/val.jsonl \
  --output_dir ./sarvam_lora_out

# evaluate the trained adapter over the 4 splits (needs adapter + full JSONL)
python src/evaluate_adapter.py \
  --adapter_dir /path/to/sarvam_lora \
  --data_dir /path/to/jsonl

# base-model baseline over the 4 splits (no adapter)
python src/evaluate_base.py \
  --data_dir /path/to/jsonl

# score any prediction JSONL (rows: prediction/target or exact/cer/wer)
python src/evaluate.py \
  --predictions /path/to/predictions.jsonl
```

`src/train_lora.py` and `vendor/punarvaani_sarvam/__init__.py` (vendored `Sarvam.py`)
implement the same training path; the vendor copy is the pinned file the run used.

## Files the full run used (private, not in this repo)

- `train.jsonl` (60,000), `val.jsonl` (4,000), `test.jsonl` (2,000) + clean counterparts
- prediction JSONLs per split (LoRA + base), used for the table above
- adapter: `adapter_config.json` + `adapter_model.safetensors` + tokenizer files
