#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate PA-Cons, HAL-Rate, and R-LEN for dialogue responses.

Definitions (matching the user’s math):

PA-Cons = (1/N) * sum_i [ 1[forall p' in P_hat: PNLI(p' -> r_hat) != CONTRADICT]
                           * 1[f_act(r) == f_act(r_hat)] ]

HAL-Rate = (1/N) * sum_i [ alpha * (100 * #unsupported_sents / #total_sents)
                           + (1 - alpha) * (100 - 100 * 1[f_act(r) == f_act(r_hat)]) ]

R-LEN = (1/N) * sum_i |r_hat|

Where:
- PNLI is RoBERTa-large-MNLI run with (premise=p', hypothesis=r_hat or each sentence).
- Unsupported sentence: a sentence in r_hat that is NOT entailed by ANY support text and/or is CONTRADICTED by ANY support text.
  We use supports = verbalized personas (P_hat) ∪ {r (gold response)}.
- f_act is a dialogue-act classifier. If you don't have a fine-tuned model, you can:
    * pass --act_clf none → we’ll use the target act label in the JSONL (pa_plus.act) as gold
      and a small heuristic on r_hat as the “prediction” (very rough baseline).
    * pass --act_clf <HF_or_local_path> → we’ll load your fine-tuned RoBERTa-large classifier.

Expected JSONL fields per row (minimally):
{
  "x": "...",                                # prompt (not used by metrics here)
  "pa_plus": {"persona": ["(s,p,o)", ...], "act": "<ActLabel>"},
  "y_plus": "[BRES] ... [ERES]",            # gold response r
  "Response": "[BRES] ... [ERES]"           # model response r_hat
}
If your prediction field is named differently, use --pred_key to point to it.
"""

import os, json, argparse, re
from typing import List, Dict, Tuple, Optional
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm
import numpy as np

# ---------------------------- Helpers ----------------------------

BRES, ERES = "[BRES]", "[ERES]"

def strip_span(text: str) -> str:
    if not text:
        return ""
    s = text.find(BRES)
    e = text.find(ERES, s + len(BRES)) if s != -1 else -1
    core = text[s + len(BRES):e].strip() if (s != -1 and e != -1) else text.strip()
    return core.strip().strip('"').strip()

def verbalize_triplet(tri: str) -> str:
    """Turn '(A, B, C)' or any loose string into a readable sentence."""
    t = (tri or "").strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    parts = [p.strip() for p in re.split(r"\s*,\s*", t) if p.strip()]
    if len(parts) == 3:
        return f"{parts[0]} {parts[1]} {parts[2]}."
    return t if t.endswith(".") else f"{t}."

def split_sentences(text: str) -> List[str]:
    # Simple, robust splitter (avoid heavyweight punkt dependency).
    # Keeps “sentences” with at least some alphabetic characters.
    cand = re.split(r"(?<=[\.\!\?])\s+", (text or "").strip())
    return [s.strip() for s in cand if any(ch.isalpha() for ch in s)]

# ---------------------------- PNLI (RoBERTa-large-MNLI) ----------------------------

class PNLI:
    def __init__(self, model_name="roberta-large-mnli", device=None, batch_size=8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text-classification",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             return_all_scores=False,
                             device=device,
                             truncation=True)
        self.batch_size = batch_size

    def infer_label(self, premise: str, hypothesis: str) -> str:
        # Labels: "ENTAILMENT" / "NEUTRAL" / "CONTRADICTION" (pipe idiosyncrasies normalized below)
        out = self.pipe({"text": premise, "text_pair": hypothesis})
        lab = out["label"].upper()
        if "ENTAIL" in lab:
            return "entailment"
        if "CONTRA" in lab:
            return "contradiction"
        return "neutral"

    def batched_labels(self, pairs: List[Tuple[str, str]]) -> List[str]:
        res = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i+self.batch_size]
            inputs = [{"text": p, "text_pair": h} for (p, h) in batch]
            outs = self.pipe(inputs)
            for o in outs:
                lab = o["label"].upper()
                if "ENTAIL" in lab:
                    res.append("entailment")
                elif "CONTRA" in lab:
                    res.append("contradiction")
                else:
                    res.append("neutral")
        return res

# ---------------------------- Dialogue Act Classifier ----------------------------

class ActClassifier:
    """
    Two modes:
      (1) HF checkpoint: a fine-tuned RoBERTa-large classifier (preferred).
      (2) Heuristic: very rough string rules if no classifier is provided.
    """
    def __init__(self, model_path: Optional[str], label_map: Optional[List[str]] = None, device=None):
        self.heuristic = False
        self.label_map = label_map
        if model_path is None or model_path.lower() == "none":
            self.heuristic = True
            return
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text-classification",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             return_all_scores=False,
                             device=device,
                             truncation=True)

        if self.label_map is None:
            # Try to deduce labels from config if present
            try:
                id2label = self.model.config.id2label
                self.label_map = [id2label[i] for i in sorted(id2label)]
            except Exception:
                self.label_map = None

    def predict(self, text: str) -> str:
        if self.heuristic:
            return self._heuristic(text)
        out = self.pipe(text)
        lab = out["label"]
        return lab

    def _heuristic(self, t: str) -> str:
        """Very rough fallback for act classification. Adjust as needed."""
        tl = (t or "").lower()
        if any(k in tl for k in ["price", "$", "cost", "total", "comes to", "amount"]):
            if any(k in tl for k in ["cannot", "not able", "no change", "fixed", "non-negotiable"]):
                return "Negotiate_Price_NoChange"
            if any(k in tl for k in ["increase", "higher", "raise"]):
                return "Negotiate_Price_Increase"
            if any(k in tl for k in ["decrease", "lower", "reduce", "discount"]):
                return "Negotiate_Price_Decrease"
            return "Tell_Price"
        if any(k in tl for k in ["add", "include", "upgrade"]):
            return "Negotiate_Add_X"
        if any(k in tl for k in ["remove", "exclude", "drop"]):
            return "Negotiate_Remove_X"
        if any(k in tl for k in ["hi", "hello", "welcome"]):
            return "Greet_Ask"
        if any(k in tl for k in ["i want", "i'd like", "i am looking", "prefer"]):
            return "Tell_Preference"
        return "Inform"

# ---------------------------- Metric Computation ----------------------------

def persona_consistency(pnli: PNLI, personas: List[str], r_hat: str) -> int:
    """
    1 if for ALL persona sentences p' we do NOT detect contradiction with r_hat; else 0.
    We run NLI with (premise=p', hypothesis=r_hat).
    """
    if not personas:
        return 1  # no personas -> vacuously true
    pairs = [(p, r_hat) for p in personas]
    labels = pnli.batched_labels(pairs)
    return 1 if all(l != "contradiction" for l in labels) else 0

def act_consistency(act_clf: ActClassifier, r_gold: str, r_hat: str,
                    gold_label_from_json: Optional[str] = None) -> int:
    """
    1 if f_act(r_gold) == f_act(r_hat).
    If no classifier is provided, use gold_label_from_json for r_gold and heuristic for r_hat.
    """
    if act_clf.heuristic:
        gold_lab = (gold_label_from_json or "").strip()
        pred_lab = act_clf.predict(r_hat)
        return 1 if gold_lab and (pred_lab == gold_lab) else 0
    else:
        g = act_clf.predict(r_gold)
        p = act_clf.predict(r_hat)
        return 1 if g == p else 0

def unsupported_rate(pnli: PNLI, supports: List[str], r_hat: str) -> float:
    """
    % of sentences in r_hat that are not entailed by ANY support, or are contradicted by ANY support.
    supports = persona verbalizations + gold response r
    """
    sents = split_sentences(r_hat)
    if not sents:
        return 100.0  # empty response => fully unsupported
    unsupported = 0
    for s in sents:
        # First check for any contradiction (strong signal)
        pairs = [(sup, s) for sup in supports]
        labels = pnli.batched_labels(pairs)
        if any(l == "contradiction" for l in labels):
            unsupported += 1
            continue
        # If no entailments from any support, count as unsupported
        if not any(l == "entailment" for l in labels):
            unsupported += 1
    return 100.0 * unsupported / max(1, len(sents))

def response_length(r_hat: str, unit: str = "words") -> int:
    core = strip_span(r_hat)
    if unit == "chars":
        return len(core)
    # default: whitespace tokens
    return len([t for t in re.split(r"\s+", core.strip()) if t])

# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True, help="JSONL with predictions & references.")
    ap.add_argument("--pred_key", default="Response", help="Key for model response (default: Response).")
    ap.add_argument("--gold_key", default="y_plus", help="Key for gold response r (default: y_plus).")
    ap.add_argument("--alpha", type=float, default=0.5, help="Alpha for HAL-Rate weighting.")
    ap.add_argument("--pnli_model", default="roberta-large-mnli", help="PNLI backbone.")
    ap.add_argument("--pnli_batch", type=int, default=8)
    ap.add_argument("--act_clf", default="none", help="HF path to your act classifier; 'none' = heuristic fallback.")
    ap.add_argument("--act_labels", nargs="*", default=None, help="Optional label list for act classifier.")
    ap.add_argument("--len_unit", choices=["words", "chars"], default="words")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    pnli = PNLI(model_name=args.pnli_model, device=device, batch_size=args.pnli_batch)
    act_clf = ActClassifier(model_path=args.act_clf, label_map=args.act_labels, device=device)

    N = 0
    pa_cons_hits = 0
    hal_terms = []
    lens = []

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scoring"):
            r = json.loads(line)

            # personas (verbalized)
            personas_raw = (r.get("pa_plus", {}) or {}).get("persona", []) or []
            personas = [verbalize_triplet(p) for p in personas_raw]

            # responses
            r_gold = strip_span(r.get(args.gold_key, ""))
            r_hat  = strip_span(r.get(args.pred_key, ""))

            if not r_hat:
                continue  # skip empty predictions

            # PA-Cons components
            persona_ok = persona_consistency(pnli, personas, r_hat)
            act_gold_label = (r.get("pa_plus", {}) or {}).get("act", "")
            act_ok = act_consistency(act_clf, r_gold, r_hat, gold_label_from_json=act_gold_label)
            pa_cons_hits += (1 if (persona_ok == 1 and act_ok == 1) else 0)

            # HAL-Rate components
            supports = personas[:]
            if r_gold:
                supports.append(r_gold if r_gold.endswith(".") else r_gold + ".")
            unsup_pct = unsupported_rate(pnli, supports, r_hat)  # [0,100]
            act_fail  = 0 if act_ok == 1 else 100
            hal_val = args.alpha * unsup_pct + (1.0 - args.alpha) * act_fail
            hal_terms.append(hal_val)

            # R-LEN
            lens.append(response_length(r_hat, unit=args.len_unit))

            N += 1

    if N == 0:
        print("No valid examples found.")
        return

    out = {
        "Count": N,
        "PA_Cons": round(pa_cons_hits / N, 6),
        "HAL_Rate": round(float(np.mean(hal_terms)), 4),
        "R_LEN_mean": round(float(np.mean(lens)), 3),
        "alpha": args.alpha,
        "len_unit": args.len_unit,
        "pnli_model": args.pnli_model,
        "act_clf": args.act_clf
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
