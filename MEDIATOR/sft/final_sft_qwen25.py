#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
import torch
import math

SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[BPER]", "[EPER]", "[BACT]", "[EACT]", "[BRES]", "[ERES]"]

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train_file", type=str, required=True)    # CSV with columns: input,output
    ap.add_argument("--valid_file", type=str, required=True)    # CSV with columns: input,output
    ap.add_argument("--output_dir", type=str, default="./qwen2.5-3b-sft")
    ap.add_argument("--seq_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--hf_token", type=str, default=None)
    return ap.parse_args()

def _to_str(x):
    # Robustly convert dataset cell to string
    if x is None:
        return ""
    # datasets may pass NaN as float('nan')
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)

def _join_io(inp, out):
    s_inp = _to_str(inp).strip()
    s_out = _to_str(out).strip()
    if s_out:
        return f"{s_inp}\n{s_out}"
    return s_inp

def formatting_func(example):
    """
    TRL calls this in batched mode (example fields are lists) OR single mode.
    Must return a string OR a list of strings of equal length to the batch.
    """
    inp = example["input"]
    out = example["output"]
    # Batched: lists
    if isinstance(inp, list):
        return [_join_io(i, o) for i, o in zip(inp, out)]
    # Single: scalars
    return _join_io(inp, out)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, token=args.hf_token)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    dtype = torch.float16 if (torch.cuda.is_available() and args.fp16) else (torch.bfloat16 if args.bf16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        token=args.hf_token,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Data
    ds = load_dataset("csv", data_files={"train": args.train_file, "validation": args.valid_file})
    train_ds = ds["train"]
    val_ds = ds["validation"]

    # LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=None
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=0.0,
        logging_steps=args.log_every,
        save_steps=args.save_every,
        evaluation_strategy="steps",
        eval_steps=args.save_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
    )

    # Trainer — no double packing
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        max_seq_length=args.seq_length,
        packing=False,  # IMPORTANT: keep False
        peft_config=lora_config,
    )

    print("➡️  Starting SFT...")
    trainer.train()

    # Save LoRA adapter
    lora_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(lora_dir, exist_ok=True)
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # Merge LoRA → full model
    print("➡️  Merging LoRA into base weights...")
    merged = PeftModel.from_pretrained(model, lora_dir)
    merged = merged.merge_and_unload()

    best_dir = os.path.join(args.output_dir, "best_merged")
    os.makedirs(best_dir, exist_ok=True)
    merged.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    print(f"✅ SFT complete. LoRA saved to: {lora_dir}")
    print(f"✅ Merged full model saved to: {best_dir}")

if __name__ == "__main__":
    main()
