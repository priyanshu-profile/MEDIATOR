#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT training for *any* HF causal LM (Qwen/LLaMA/Mistral/etc.) with LoRA + merge.

Example:
CUDA_VISIBLE_DEVICES=0 python3 sft_any.py \
  --raw_csv ABN_final_dataset_with_triplets_by_conv.csv \
  --prep_dir ./prep_sft \
  --output_dir ./qwen2.5-3b-sft \
  --model_path Qwen/Qwen2.5-3B-Instruct \
  --seq_length 2048 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_steps 1000 \
  --lr_scheduler_type cosine \
  --fp16 \
  --gradient_checkpointing
"""

import os
import ast
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable

import math
import random
import pandas as pd

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer

# LoRA / PEFT
from peft import LoraConfig, PeftModel

# ----------------------------
# Config / Constants
# ----------------------------
SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[BPER]", "[EPER]", "[BACT]", "[EACT]", "[BRES]", "[ERES]"]
NEUTRAL_AGENT_ACT = "general_response"

INST = (
    "You are a helpful travel booking assistant. Given the dialogue history between an agent and a traveler, "
    "and the traveler's current turn, produce: (i) the traveler's relevant persona triplets, (ii) the agent's dialogue act, "
    "and (iii) the agent's next response. Use the special tokens exactly as specified."
)

# ----------------------------
# Utilities for persona parsing
# ----------------------------
def safe_parse_list(x):
    if pd.isna(x) or x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    return []

def build_conversations(df: pd.DataFrame) -> Dict[Any, List[Dict[str, Any]]]:
    grouped = {}
    for conv_id, sub in df.groupby("conv_id", sort=True):
        records = sub.to_dict("records")
        grouped[conv_id] = records
    return grouped

def format_turn(agent_act: Optional[str], agent_utt: Optional[str], traveler_utt: Optional[str]) -> str:
    if traveler_utt is not None:
        return traveler_utt.strip()
    act = agent_act.strip() if (agent_act and agent_act.strip()) else NEUTRAL_AGENT_ACT
    utt = (agent_utt or "").strip()
    return f"[BACT] {act} [EACT] {utt}".strip()

def build_samples_for_conv(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build (input, output) pairs for SFT.
    Input X = INST + history (act-tagged) + current traveler turn
    Output Y = [BOS] [BPER] persona... [EPER] [BACT] act [EACT] [BRES] agent_resp [ERES] [EOS]
    """
    if not records:
        return []

    first = records[0]
    persona_triplets = safe_parse_list(first.get("persona_triplets", []))

    # Collect ordered turns (role = agent/traveler)
    turns = []
    for r in records:
        # Speaker: 1 = agent, 2 = traveler (per your example)
        spk = int(r.get("Speaker"))
        utt = str(r.get("Utterance", "")).strip()
        if spk == 1:
            turns.append({"role": "agent", "text": utt})
        else:
            turns.append({"role": "traveler", "text": utt})

    samples = []
    history = []

    for i, t in enumerate(turns):
        if t["role"] == "agent":
            # push agent turn with a neutral act tag (or you can map real acts if you have them)
            history.append(format_turn(agent_act=NEUTRAL_AGENT_ACT, agent_utt=t["text"], traveler_utt=None))
        else:
            # when traveler is followed by an agent reply, create a (X,Y) pair
            if i + 1 < len(turns) and turns[i + 1]["role"] == "agent":
                H = "\n".join(history) if history else ""
                T_t = t["text"]
                X = INST + "\n\n" + (H + "\n" if H else "") + T_t

                A_t = turns[i + 1]["text"]
                if persona_triplets:
                    per_block = " ".join(persona_triplets)
                else:
                    per_block = ""

                Y = (
                    "[BOS] "
                    + "[BPER] "
                    + (per_block + " " if per_block else "")
                    + "[EPER] "
                    + "[BACT] "
                    + NEUTRAL_AGENT_ACT
                    + " [EACT] "
                    + "[BRES] "
                    + A_t
                    + " [ERES] "
                    + "[EOS]"
                )
                samples.append({"input": X, "output": Y})

            # always append the raw traveler utterance to history
            history.append(format_turn(agent_act=None, agent_utt=None, traveler_utt=t["text"]))

    return samples

def build_and_write_splits(raw_csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(
        raw_csv_path,
        low_memory=False,
        dtype={"persona_evidences": "string", "persona_triplets": "string"}
    )

    # Required columns
    req = {"conv_id", "Speaker", "Utterance", "persona_evidences", "persona_triplets"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    grouped = build_conversations(df)

    # pick conv_ids that *start* with persona_triplets (as you described)
    annotated_convs = []
    for cid, conv in grouped.items():
        if not conv:
            continue
        first = conv[0]
        pt = safe_parse_list(first.get("persona_triplets"))
        if pt:
            annotated_convs.append(cid)

    total = len(annotated_convs)
    print(f"✅ Found {total} conversations with persona triplets at the start.")

    annotated_convs = sorted(annotated_convs)

    # 50/25/25 split by conversation
    n_train = total // 2
    n_val = (total - n_train) // 2
    n_test = total - n_train - n_val

    conv_splits = {
        "train": annotated_convs[:n_train],
        "val": annotated_convs[n_train:n_train + n_val],
        "test": annotated_convs[n_train + n_val:]
    }

    def collect(conv_list):
        out = []
        for cid in conv_list:
            out.extend(build_samples_for_conv(grouped[cid]))
        return out

    train_samples = collect(conv_splits["train"])
    val_samples   = collect(conv_splits["val"])
    test_samples  = collect(conv_splits["test"])

    # Save CSVs
    pd.DataFrame(train_samples).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    pd.DataFrame(val_samples).to_csv(os.path.join(out_dir, "val.csv"), index=False)
    pd.DataFrame(test_samples).to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print("\n📊 Split summary:")
    print(f"  train: {len(train_samples)} samples from {len(conv_splits['train'])} conv_ids")
    print(f"  val:   {len(val_samples)} samples from {len(conv_splits['val'])} conv_ids")
    print(f"  test:  {len(test_samples)} samples from {len(conv_splits['test'])} conv_ids")

    return (
        os.path.join(out_dir, "train.csv"),
        os.path.join(out_dir, "val.csv"),
        os.path.join(out_dir, "test.csv"),
    )

# ----------------------------
# Packing helpers
# ----------------------------
def prepare_sample_text(example):
    # Prompt + target in one stream; we pack multiple of these later.
    return f"{example['input']}\n{example['output']}".strip()

def chars_token_ratio(dataset, tokenizer, num_samples=400):
    # Estimate char/token ratio for packing
    num_samples = min(num_samples, len(dataset))
    samples = dataset.select(range(num_samples))
    total_chars = 0
    total_tokens = 0
    for ex in samples:
        s = prepare_sample_text(ex)
        total_chars += len(s)
        total_tokens += len(tokenizer(s, add_special_tokens=False)["input_ids"])
    return (total_chars / max(1, total_tokens)) if total_tokens else 3.0

class ConstantLengthDataset(torch.utils.data.IterableDataset):
    """
    Packs multiple texts into sequences of length `seq_length` (approx via chars->tokens ratio).
    Returns tokenized tensors directly (input_ids, labels).
    """
    def __init__(
        self,
        tokenizer,
        dataset,
        formatting_func,
        infinite=False,
        seq_length=2048,
        chars_per_token=3.6,
        eos_token_id=None,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.formatting_func = formatting_func
        self.infinite = infinite
        self.seq_length = seq_length
        self.chars_per_token = chars_per_token
        self.eos_token_id = eos_token_id or tokenizer.eos_token_id

    def _sample_iterator(self):
        idx = 0
        n = len(self.dataset)
        while True:
            if idx >= n:
                if self.infinite:
                    idx = 0
                    continue
                else:
                    break
            yield self.dataset[idx]
            idx += 1

    def __iter__(self):
        buffer, buffer_len = [], 0
        approx_char_limit = int(self.seq_length * self.chars_per_token)

        for ex in self._sample_iterator():
            s = self.formatting_func(ex)
            buffer.append(s)
            buffer_len += len(s)

            if buffer_len > approx_char_limit:
                tokenized = self.tokenizer("\n\n".join(buffer), add_special_tokens=False)["input_ids"]
                buffer, buffer_len = [], 0

                for i in range(0, len(tokenized), self.seq_length):
                    input_ids = tokenized[i : i + self.seq_length]
                    if len(input_ids) == 0:
                        continue
                    if len(input_ids) < self.seq_length:
                        input_ids = input_ids + [self.eos_token_id] * (self.seq_length - len(input_ids))
                    ii = torch.tensor(input_ids, dtype=torch.long)
                    yield {"input_ids": ii, "labels": ii.clone()}

        # flush tail
        if buffer:
            tokenized = self.tokenizer("\n\n".join(buffer), add_special_tokens=False)["input_ids"]
            for i in range(0, len(tokenized), self.seq_length):
                input_ids = tokenized[i : i + self.seq_length]
                if len(input_ids) == 0:
                    continue
                if len(input_ids) < self.seq_length:
                    input_ids = input_ids + [self.eos_token_id] * (self.seq_length - len(input_ids))
                ii = torch.tensor(input_ids, dtype=torch.long)
                yield {"input_ids": ii, "labels": ii.clone()}

def create_sft_datasets(tokenizer, args):
    # Full splits by default; use --fast_test to speed up
    train_split = "train[:5]" if args.fast_test else "train"
    valid_split = "validation[:5]" if args.fast_test else "validation"

    train_data = load_dataset("csv", data_files={"train": [args.train_file]}, split=train_split)
    valid_data = load_dataset("csv", data_files={"validation": [args.valid_file]}, split=valid_split)

    print(f"Train rows: {len(train_data)}, Validation rows: {len(valid_data)}")

    cpt = chars_token_ratio(train_data, tokenizer, min(400, len(train_data)))
    print(f"Estimated char/token ratio: {cpt:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=cpt,
        eos_token_id=tokenizer.eos_token_id,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=cpt,
        eos_token_id=tokenizer.eos_token_id,
    )
    return train_dataset, valid_dataset

def run_sft(args, tokenizer):
    # LoRA config
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"
    )

    # Base model
    dtype = (
        torch.float16 if (args.fp16 and torch.cuda.is_available()) else
        (torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        token=getattr(args, "hf_token", None),
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Add special tokens + set pad token
    _ = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    # Datasets
    train_dataset, eval_dataset = create_sft_datasets(tokenizer, args)

    max_steps = 3 if args.fast_test else args.max_steps
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        max_steps=max_steps,
        weight_decay=args.weight_decay,
        logging_steps=1 if args.fast_test else args.log_freq,
        save_steps=999999 if args.fast_test else args.save_freq,
        save_strategy="epoch",
        eval_strategy="epoch",   # <- correct key
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # Important: since we already feed tokenized tensors (input_ids, labels),
    # set packing=False and let TRL skip its internal formatting/tokenization.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        packing=False,   # <-- we already packed in ConstantLengthDataset
    )

    print("SFT training...")
    trainer.train()

    # --- Save & Merge LoRA ---
    lora_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(lora_dir, exist_ok=True)
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    print("Merging LoRA weights into full model...")
    merged = PeftModel.from_pretrained(model, lora_dir)
    merged = merged.merge_and_unload()

    best_model_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    merged.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print(f"SFT completed ✅ | Merged model saved at: {best_model_dir}")

# ----------------------------
# CLI
# ----------------------------
def get_args():
    p = argparse.ArgumentParser()
    # raw csv & preprocessing output
    p.add_argument("--raw_csv", type=str, required=True, help="Path to your original dataset CSV")
    p.add_argument("--prep_dir", type=str, default="./prep_sft", help="Where to write preprocessed CSV splits")

    # model & tokenizer I/O
    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir", type=str, default="./sft_out")
    p.add_argument("--hf_token", type=str, default=None)

    # SFT data files (filled automatically after preprocessing)
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--valid_file", type=str, default=None)
    p.add_argument("--test_file", type=str, default=None)

    # training hparams
    p.add_argument("--seq_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--num_warmup_steps", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--save_freq", type=int, default=200)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # precision
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    # quick smoke test
    p.add_argument("--fast_test", action="store_true")
    return p.parse_args()

def main():
    args = get_args()

    # Step 1: Build & write splits from raw CSV
    train_csv, val_csv, test_csv = build_and_write_splits(args.raw_csv, args.prep_dir)

    # Fill args for SFT loader that uses load_dataset("csv", ...)
    args.train_file = train_csv
    args.valid_file = val_csv
    args.test_file = test_csv

    # Step 2: Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 3: Run SFT (trainer + LoRA + merge)
    run_sft(args, tokenizer)

if __name__ == "__main__":
    main()

    
