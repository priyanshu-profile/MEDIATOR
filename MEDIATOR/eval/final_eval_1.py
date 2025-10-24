#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate generations with automatic metrics and heuristic goal-attainment scores.

INPUT
------
Each model's generations are in a JSONL file with records like your example:
{
  "x": "...",
  "pa_plus": {"persona": [...], "act": "Inform"},
  "y_plus": "[BRES] ... [ERES]",            # gold
  "PromptUsed": "...[BRES]",
  "Response":  "[BRES] generated ... [ERES]"
  # (other fields are preserved but not required)
}

You can pass multiple files like:
  --file pa_dpo=outputs/pa_dpo.jsonl --file dpo=outputs/dpo.jsonl

REQUIREMENTS
------------
pip install torch transformers sacrebleu bert-score sentence-transformers pandas numpy

USAGE
-----
python3 scripts/eval_scores.py \
  --ref_lm_dir final_qwen_sft/best_merged \
  --file pa_dpo=final_outputs/pa_dpo.jsonl \
  --file dpo=outputs/dpo.jsonl \
  --file kto=outputs/kto.jsonl \
  --out_dir eval_out \
  --pa_tau 0.35 --act_threshold 0.5 --ppl_conditional

Outputs:
- eval_out/<model_name>_per_sample.jsonl
- eval_out/<model_name>_summary.json
- eval_out/summary_table.csv (all models)
"""

import os, re, json, argparse, math, statistics
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer, util as sbert_util

# ------------------------ helpers: spans & cleaning ------------------------

def extract_span(text: str) -> str:
    """Return content between [BRES] ... [ERES]. If not found, return stripped text."""
    if text is None:
        return ""
    s = text.find("[BRES]")
    e = text.find("[ERES]", s + 6) if s != -1 else -1
    if s != -1 and e != -1:
        return text[s+6:e].strip().strip('"').strip()
    return text.strip().strip('"').strip()

def word_tokens(s: str) -> List[str]:
    # very lightweight tokenization
    return re.findall(r"[A-Za-z0-9$€¥£]+(?:['-][A-Za-z0-9]+)?", s)

def ngrams(seq: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(seq[i:i+n]) for i in range(0, max(0, len(seq)-n+1))]

def safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if np.isfinite(x)]
    return float(sum(xs) / max(1, len(xs)))

# ------------------------ PPL under a reference LM ------------------------

class RefScorer:
    """
    Computes token-level NLL and PPL for a response span, optionally conditioned on PromptUsed.
    """
    def __init__(self, lm_dir: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(lm_dir, use_fast=False, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else (torch.float16 if torch.cuda.is_available() else torch.float32))
        self.model = AutoModelForCausalLM.from_pretrained(
            lm_dir, trust_remote_code=True, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def ppl_for_record(self, prompt_used: str, response_span: str, conditional: bool = True) -> Tuple[float, float]:
        """
        Returns (nll, ppl) for the response tokens only.
        conditional=True: score response given prompt (labels=-100 on prompt).
        conditional=False: score response standalone.
        """
        if not response_span:
            return float("nan"), float("nan")
        if conditional:
            # concatenate but mask the prompt
            full = (prompt_used or "").rstrip() + "\n" + "[BRES] " + response_span + " [ERES]"
            enc = self.tok(full, add_special_tokens=False, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)

            # find the [BRES] position
            bres_ids = self.tok.convert_tokens_to_ids("[BRES]")
            ids_list = input_ids[0].tolist()
            try:
                # first match of [BRES]
                start_idx = ids_list.index(bres_ids)
            except ValueError:
                start_idx = max(0, input_ids.shape[1] - 64)  # fallback: last chunk

            labels = input_ids.clone()
            labels[:, :start_idx+1] = -100  # ignore everything up to and including [BRES]
        else:
            full = response_span
            enc = self.tok(full, add_special_tokens=False, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            labels = input_ids.clone()

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[:, :-1, :]
            labels_shift = labels[:, 1:]
            attn_shift = attn[:, 1:]
            valid = (labels_shift != -100) & (attn_shift == 1)
            logp = torch.log_softmax(logits, dim=-1)
            gathered = torch.gather(logp, -1, labels_shift.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
            tok_logp = gathered[valid]
            if tok_logp.numel() == 0:
                return float("nan"), float("nan")
            nll = -tok_logp.mean().item()  # mean per-token NLL (nat log)
            ppl = math.exp(nll)
            return nll, ppl

# ------------------------ Dialogue act heuristics ------------------------

ACT_PATTERNS = {
    "Inform": [
        r"\b(the (total )?cost|price|comes to|is priced at|it includes|you'll get)\b",
        r"\b(we offer|package includes|itinerary)\b",
    ],
    "Ask_Price": [r"\b(how much|what'?s the price|cost\?)\b"],
    "Tell_Price": [r"\b(total cost|comes out to|price is|it is \$?\d)"],
    "Negotiate_Price_Increase": [
        r"\b(counter (offer|proposal)|propose[d]?|suggest(ed)?\b).{0,40}\$\d",
        r"\b(need to|have to)\s+(increase|raise|go up)\b",
    ],
    "Negotiate_Price_Decrease": [
        r"\b(could you|can you|would you)\s+(lower|reduce|drop)\b",
        r"\b(closer to|around)\s*\$?\d",
    ],
    "Negotiate_Price_NoChange": [
        r"\b(no(t)?\s+lower|cannot (go|drop) below|price is final|non-?negotiable|cannot reduce)\b",
    ],
    "Negotiate_Remove_X": [
        r"\b(remove|drop|exclude|not interested in)\b.*\b(tour|activity|add-?on|amenit(y|ies))\b",
    ],
    "Negotiate_Remove_Y_Add_X": [
        r"\b(remove|drop|exclude)\b.*\b(add|include|replace with)\b",
    ],
    "Greet_Ask": [r"\b(welcome|hi|hello|how can i help)\b"],
}

def predict_act(text: str) -> str:
    t = text.lower()
    for act, pats in ACT_PATTERNS.items():
        for p in pats:
            if re.search(p, t):
                return act
    # default fallback
    return "Inform" if re.search(r"\b(price|include|package|itinerary)\b", t) else "General"

# ------------------------ Persona alignment & hallucination heuristics ------------------------

def persona_keywords(persona_list: List[str]) -> Dict[str, bool]:
    joined = " ".join(persona_list).lower()
    return {
        "budget": any(k in joined for k in ["budget", "afford", "price", "negotiat", "value for money"]),
        "avoid_photo": ("photo" in joined and any(w in joined for w in ["avoid","not interested","decline"])),
        "seek_luxury": any(k in joined for k in ["luxury","luxurious","high-end","premium"]),
        "seeks_custom": any(k in joined for k in ["custom", "tailor", "remove", "adjust"]),
    }

def detect_hallucination(text: str, pk: Dict[str,bool], target_act: str) -> Tuple[bool, str]:
    t = text.lower()
    # Artifact checks
    if any(tag in text for tag in ["[BOS]", "[EOS]", "[BPER]", "[EPER]", "[BACT]", "[EACT]", "<|im_start|>", "vinfos"]):
        return True, "meta-artifact"
    if re.search(r"[\uFFFD]|[\u0378-\u0379\u0380-\u0383]", text):  # � replacement or weird codepoints
        return True, "unicode-garbage"

    # Persona contradictions
    if pk.get("budget") and re.search(r"\b(non-?negotiable|fixed price|cannot reduce|no discount)\b", t):
        return True, "persona-contradiction-budget"
    if pk.get("avoid_photo") and re.search(r"\b(photo(graphy)? tour|photo session)\b", t):
        return True, "persona-contradiction-photo"
    if not pk.get("seek_luxury") and re.search(r"\b(luxury only|exclusive upgrade|signature tier)\b", t):
        return True, "persona-contradiction-luxury"

    # Act mismatch is handled in PA-Cons but also counts toward hallucination.
    pred_act = predict_act(text)
    if target_act and pred_act != target_act:
        return True, "act-mismatch"

    return False, ""

# ------------------------ main eval ------------------------

def evaluate_file(
    name: str,
    path: str,
    ref_scorer: RefScorer,
    sbert: SentenceTransformer,
    pa_tau: float = 0.35,
    ppl_conditional: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{name}: empty file")

    # Prepare corpora for BLEU/BERTScore/Distinct
    gens = [extract_span(r.get("Response", "")) for r in rows]
    refs = [extract_span(r.get("y_plus", "")) for r in rows]

    # BLEU-4
    bleu = sacrebleu.corpus_bleu(gens, [refs], smooth_method="exp", force=True).score

    # BERTScore-F1
    P, R, F1 = bertscore(gens, refs, lang="en", verbose=False)
    bs_f1_corpus = float(F1.mean().item())

    # Distinct-3 (corpus-level)
    all_3 = []
    for g in gens:
        toks = word_tokens(g.lower())
        all_3.extend(ngrams(toks, 3))
    uniq_3 = len(set(all_3))
    d3 = uniq_3 / max(1, len(all_3))

    # Response length
    r_lens = [len(word_tokens(g)) for g in gens]
    r_len_mean = safe_mean([float(x) for x in r_lens])

    # SBERT for persona alignment
    pa_cons_flags = []
    hal_flags = []
    per_sample = []

    # Pre-embed persona strings to speed up
    persona_texts = []
    for r in rows:
        pa = r.get("pa_plus") or {}
        persona_list = pa.get("persona", []) or []
        act = pa.get("act", "")
        persona_texts.append((" ".join(persona_list) + f" Act: {act}").strip())
    emb_pa = sbert.encode(persona_texts, normalize_embeddings=True, convert_to_numpy=True)
    emb_gen = sbert.encode(gens, normalize_embeddings=True, convert_to_numpy=True)

    for i, r in enumerate(rows):
        gen_span = gens[i]
        ref_span = refs[i]
        prompt_used = r.get("PromptUsed", r.get("x", ""))

        # PPL over response tokens only
        nll, ppl = ref_scorer.ppl_for_record(prompt_used, gen_span, conditional=ppl_conditional)

        # Act prediction & PA consistency
        target_act = (r.get("pa_plus") or {}).get("act", "")
        pred_act = predict_act(gen_span)
        act_ok = (pred_act == target_act) if target_act else True

        # Persona alignment via cosine sim
        sim = float(np.dot(emb_pa[i], emb_gen[i]))  # embeddings normalized
        persona_ok = (sim >= pa_tau)

        pa_cons = (1 if (act_ok and persona_ok) else 0)
        pa_cons_flags.append(pa_cons)

        # Hallucination heuristic
        pk = persona_keywords((r.get("pa_plus") or {}).get("persona", []) or [])
        hallu, hallu_reason = detect_hallucination(gen_span, pk, target_act)
        hal_flags.append(1 if hallu else 0)

        per_sample.append({
            "x": r.get("x", "")[:3000],
            "target_act": target_act,
            "pred_act": pred_act,
            "persona_sim": round(sim, 4),
            "persona_ok": bool(persona_ok),
            "act_ok": bool(act_ok),
            "PA_Cons": int(pa_cons),
            "HAL_Flag": int(hallu),
            "HAL_Reason": hallu_reason,
            "PPL": float(ppl) if np.isfinite(ppl) else None,
            "NLL": float(nll) if np.isfinite(nll) else None,
            "BLEU4": None,          # (global)
            "BERTScore_F1": None,   # (global)
            "Distinct3": None,      # (global)
            "RespLen": int(len(word_tokens(gen_span))),
            "RefSpan": ref_span,
            "GenSpan": gen_span,
        })

    # Summaries
    ppl_vals = [ps["PPL"] for ps in per_sample if ps["PPL"] is not None]
    summary = {
        "Model": name,
        "Files": path,
        "Count": len(rows),
        "BLEU4": round(bleu, 3),
        "BERTScore_F1": round(bs_f1_corpus, 4),
        "Distinct3": round(d3, 4),
        "RespLen_mean": round(r_len_mean, 2),
        "PPL_mean": round(safe_mean(ppl_vals), 3) if ppl_vals else None,
        "PA_Cons_rate": round(sum(pa_cons_flags) / max(1, len(pa_cons_flags)), 4),
        "HAL_Rate": round(sum(hal_flags) / max(1, len(hal_flags)), 4),
        "PA_threshold": pa_tau,
        "PPL_conditional": bool(ppl_conditional),
    }

    # Attach global scores to first row for reference (optional)
    for ps in per_sample:
        ps["BLEU4_global"] = summary["BLEU4"]
        ps["BERTScore_F1_global"] = summary["BERTScore_F1"]
        ps["Distinct3_global"] = summary["Distinct3"]

    df = pd.DataFrame(per_sample)
    return df, summary

# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_lm_dir", required=True, help="Reference LM for PPL (e.g., your SFT best_merged).")
    ap.add_argument("--file", action="append", default=[],
                    help="Named file arg: name=path/to/jsonl (repeatable).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pa_tau", type=float, default=0.35, help="SBERT cosine threshold for persona alignment.")
    ap.add_argument("--ppl_conditional", action="store_true", help="If set, score PPL conditioned on PromptUsed.")
    ap.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    if not args.file:
        raise SystemExit("Provide at least one --file name=path")

    os.makedirs(args.out_dir, exist_ok=True)
    ref = RefScorer(args.ref_lm_dir)
    sbert = SentenceTransformer(args.sbert_model)

    all_summaries = []
    for spec in args.file:
        try:
            name, path = spec.split("=", 1)
        except ValueError:
            raise SystemExit(f"--file expects name=path, got: {spec}")
        df, summ = evaluate_file(name, path, ref, sbert, pa_tau=args.pa_tau, ppl_conditional=args.ppl_conditional)

        per_sample_fp = os.path.join(args.out_dir, f"{name}_per_sample.jsonl")
        with open(per_sample_fp, "w", encoding="utf-8") as f:
            for rec in df.to_dict("records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary_fp = os.path.join(args.out_dir, f"{name}_summary.json")
        with open(summary_fp, "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)
        print(f"[{name}] wrote {per_sample_fp} and {summary_fp}")
        all_summaries.append(summ)

    # combined CSV
    table = pd.DataFrame(all_summaries)
    table_path = os.path.join(args.out_dir, "summary_table.csv")
    table.to_csv(table_path, index=False)
    print(f"[ALL] summary table → {table_path}")

if __name__ == "__main__":
    main()
