#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, random
from typing import Dict, Any, List
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
try:
    from peft import LoraConfig
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

SPECIAL_TOKENS = ["[BOS]","[EOS]","[BPER]","[EPER]","[BACT]","[EACT]","[BRES]","[ERES]"]

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def add_special_tokens(tok):
    add = [t for t in SPECIAL_TOKENS if t not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    tok.padding_side = "left"
    return tok

def span(text: str) -> str:
    # return inner of [BRES] ... [ERES]; if missing, return text
    t = (text or "").strip()
    s = t.find("[BRES]")
    e = t.find("[ERES]", s+6) if s != -1 else -1
    core = t[s+6:e].strip() if (s != -1 and e != -1) else t
    # strip outer quotes if any
    core = core.strip().strip('"').strip()
    return core

def load_pref_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            x = str(r.get("x","")).strip()
            yp = span(r.get("y_plus",""))
            ym = span(r.get("y_minus",""))
            if not x or not yp or not ym:
                continue
            rows.append({"prompt": x, "chosen": yp, "rejected": ym})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_model_dir", required=True)
    ap.add_argument("--train_jsonl",   required=True)
    ap.add_argument("--output_dir",    required=True)
    # lengths
    ap.add_argument("--max_length",        type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=768)
    ap.add_argument("--max_target_length", type=int, default=256)
    # dpo
    ap.add_argument("--beta",   type=float, default=0.1)
    # train
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--warmup_ratio",  type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    # precision
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    # LoRA
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", nargs="*", default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build HF dataset in TRL expected shape
    data = load_pref_jsonl(args.train_jsonl)
    ds = Dataset.from_list(data)

    # Load model/tokenizer
    tok = AutoTokenizer.from_pretrained(args.sft_model_dir, use_fast=False, trust_remote_code=True)
    tok = add_special_tokens(tok)

    policy = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir, trust_remote_code=True
    )
    policy.resize_token_embeddings(len(tok))

    # Optional: gradient checkpointing (small models often fine without)
    policy.config.use_cache = False

    # DPO config
    dcfg = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16, fp16=args.fp16,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        #max_target_length=args.max_target_length,
        beta=args.beta,
        warmup_ratio=args.warmup_ratio,
        report_to="none",
        remove_unused_columns=False,
    )

    # Optional LoRA
    peft_cfg = None
    if args.use_lora:
        assert _HAS_PEFT, "peft not installed"
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets, bias="none", task_type="CAUSAL_LM"
        )

    # Let TRL clone a frozen reference model automatically (ref_model=None)
    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        args=dcfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("✅ Vanilla DPO done →", args.output_dir)

if __name__ == "__main__":
    main()
