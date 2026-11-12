"""LoRA fine-tuning of sarvamai/sarvam-1 for Hindi phoneme-to-text.

Expects JSONL rows: {"prompt": "<S2S>\\n<PHONEMES>\\n...\\n</PHONEMES>\\n<TEXT>\\n", "target": "..."}.

The four prompt tags are registered as new single tokens; their embedding rows are
saved with the adapter via modules_to_save so inference sees the trained vectors.
"""

import argparse
import inspect

import torch

try:
    from data_utils import TAGS
except ImportError:  # allow `python -m src.train_lora` from the repo root
    from src.data_utils import TAGS


def train_sarvam_lora(
    train_jsonl_path: str,
    val_jsonl_path: str,
    output_dir: str = "./sarvam_lora_out",
    base_model_id: str = "sarvamai/sarvam-1",
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 2e-4,
    max_length: int = 512,
    gradient_accumulation_steps: int = 1,
    eval_steps: int = 500,
    resume_from_checkpoint=None,
    bf16=None,
):
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

    if bf16 is None:
        bf16 = bool(
            torch.cuda.is_available()
            and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
    use_fp16 = bool(torch.cuda.is_available() and not bf16)
    print(f"[train] precision: {'bf16' if bf16 else ('fp16' if use_fp16 else 'fp32')}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": TAGS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files={"train": train_jsonl_path, "val": val_jsonl_path})

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    def tokenize_fn(batch):
        input_ids, attention_mask, labels = [], [], []
        for prompt, target in zip(batch["prompt"], batch["target"]):
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [eos_id]
            ids = prompt_ids + target_ids
            lab = [-100] * len(prompt_ids) + target_ids
            if len(ids) > max_length:
                ids = ids[:max_length]
                lab = lab[:max_length]
            pad = max_length - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    if torch.cuda.is_available():
        model.to("cuda")
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=use_fp16,
        bf16=bf16,
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=5,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        report_to="none",
    )
    training_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_arg = "eval_strategy" if "eval_strategy" in training_params else "evaluation_strategy"
    training_kwargs[eval_arg] = "steps"
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = dict(model=model, args=training_args,
                          train_dataset=tokenized["train"], eval_dataset=tokenized["val"])
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    else:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train] adapter saved to {output_dir}")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sarvam-1 with LoRA")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--output_dir", default="./sarvam_lora_out")
    parser.add_argument("--base_model", default="sarvamai/sarvam-1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    args = parser.parse_args()
    train_sarvam_lora(
        args.train_jsonl, args.val_jsonl, args.output_dir, args.base_model,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        max_length=args.max_length, gradient_accumulation_steps=args.grad_accum,
        eval_steps=args.eval_steps,
    )


if __name__ == "__main__":
    main()
