# train_act_classifier.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_jsonl_datasets(train_path, valid_path):
    ds = load_dataset("json", data_files={"train": train_path, "validation": valid_path})
    labels = sorted(set([r["label"] for r in ds["train"]]))
    label2id = {lab:i for i,lab in enumerate(labels)}
    id2label = {i:lab for lab,i in label2id.items()}

    def _prep(ex):
        return {"text": ex["text"], "labels": label2id[ex["label"]]}
    ds = ds.map(_prep, remove_columns=[c for c in ds["train"].column_names if c not in ["text","labels"]])
    return ds, label2id, id2label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", default="pact_act_data/train.jsonl")
    ap.add_argument("--valid_file", default="pact_act_data/valid.jsonl")
    ap.add_argument("--model_name", default="roberta-base")  # use roberta-large if you want stronger
    ap.add_argument("--output_dir", default="ckpts/act_roberta_base")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    ds, label2id, id2label = load_jsonl_datasets(args.train_file, args.valid_file)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    def tok_fn(batch): return tok(batch["text"], truncation=True, max_length=384)
    ds_tok = ds.map(tok_fn, batched=True)
    collator = DataCollatorWithPadding(tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), label2id=label2id, id2label=id2label
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        acc = accuracy_score(p.label_ids, preds)
        f1  = f1_score(p.label_ids, preds, average="macro")
        return {"acc": acc, "macro_f1": f1}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print("✅ Saved act classifier to", args.output_dir)

if __name__ == "__main__":
    main()
