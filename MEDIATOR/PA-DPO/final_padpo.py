#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PA-DPO training (end-to-end, single file)

- Policy is initialized from an SFT checkpoint (same dir also used as frozen ref).
- Optimizes: L = DPO(x, y+, y-) + lambda_pa * DPO(x ⊕ pa+, y+) vs (x ⊕ pa-, y-)
- Stable numerics: mean logprobs (length-normalized), fp32 advantages, clamped logits.
- Optional LoRA so you don't train all 3B params.

Expected JSONL fields per row:
{
  "x": "<prompt string>",
  "y_plus": "[BRES] ... [ERES]",
  "y_minus": "[BRES] ... [ERES]",
  "pa_plus":  {"persona": ["(subj, pred, obj)", ...], "act": "Negotiate_Price_..."},
  "pa_minus": {"persona": ["..."], "act": "..." }
}
"""

import os, math, json, argparse, random, time, csv
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)

# ---------------- Special tokens ----------------
BOS, EOS = "[BOS]","[EOS]"
BPER, EPER = "[BPER]","[EPER]"
BACT, EACT = "[BACT]","[EACT]"
BRES, ERES = "[BRES]","[ERES]"
SPECIAL_TOKENS = [BOS, EOS, BPER, EPER, BACT, EACT, BRES, ERES]

# ---------------- Utilities ----------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def add_special_tokens(tokenizer):
    add = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if add: tokenizer.add_special_tokens({"additional_special_tokens": add})
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    return tokenizer

def wrap_reply(text: str) -> str:
    s = (text or "").strip().strip('"')
    if not s.startswith(BRES): s = f"{BRES} {s}"
    if not s.endswith(ERES):
        if not s.endswith((".", "!", "?")): s += "."
        s += f" {ERES}"
    return s

def _as_triplet_str(x):
    if isinstance(x, dict):
        s = x.get("s") or x.get("subj") or x.get("subject") or x.get("head") or x.get("source")
        p = x.get("p") or x.get("pred") or x.get("predicate") or x.get("rel") or x.get("relation")
        o = x.get("o") or x.get("obj") or x.get("object") or x.get("tail") or x.get("target")
        if any([s,p,o]): return f"({str(s or '').strip()}, {str(p or '').strip()}, {str(o or '').strip()})"
        return str(x)
    if isinstance(x, (list, tuple)):
        if len(x) == 3 and all(not isinstance(t, (list, dict)) for t in x):
            return f"({str(x[0]).strip()}, {str(x[1]).strip()}, {str(x[2]).strip()})"
        return " ".join(_as_triplet_str(t) for t in x if t is not None)
    return str(x)

def _normalize_persona_list(persona):
    out = []
    def walk(z):
        if z is None: return
        if isinstance(z, str):
            zs = z.strip()
            if zs: out.append(zs)
        elif isinstance(z, dict):
            out.append(_as_triplet_str(z))
        elif isinstance(z, (list, tuple)):
            for t in z: walk(t)
        else:
            out.append(str(z))
    walk(persona)
    return out

def _norm_pa(pa_obj):
    if isinstance(pa_obj, str):
        try: pa_obj = json.loads(pa_obj)
        except Exception: pa_obj = {"persona":[pa_obj], "act":""}
    if not isinstance(pa_obj, dict): pa_obj = {}
    persona = _normalize_persona_list(pa_obj.get("persona", []))
    act = pa_obj.get("act", "")
    if not isinstance(act, str): act = str(act)
    return {"persona": persona, "act": act}

def serialize_pa(persona_list, act_label):
    persona_flat = _normalize_persona_list(persona_list)
    persona_str = " ".join(p.strip() for p in persona_flat if p and p.strip())
    act_label = (act_label or "general_response").strip()
    return f"{BPER} {persona_str} {EPER} {BACT} {act_label} {EACT}"

# ---------------- Dataset ----------------
class PADPODataset(Dataset):
    def __init__(self, jsonl_path: str):
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                x = str(r.get("x",""))
                if not x.strip(): continue
                y_p = wrap_reply(r.get("y_plus",""))
                y_m = wrap_reply(r.get("y_minus",""))
                pa_p = _norm_pa(r.get("pa_plus", {}))
                pa_m = _norm_pa(r.get("pa_minus", {}))
                if not y_p.strip() or not y_m.strip():  # basic filter
                    continue
                rows.append({"x": x, "y_plus": y_p, "y_minus": y_m, "pa_plus": pa_p, "pa_minus": pa_m})
        self.data = rows
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

# ---------------- Tokenization / logprobs ----------------
def concat_and_tokenize(tokenizer, prompts: List[str], replies: List[str],
                        max_prompt_len: int, max_target_len: int):
    p_ids = tokenizer(prompts, add_special_tokens=False, padding=False,
                      truncation=True, max_length=max_prompt_len)
    r_ids = tokenizer(replies,  add_special_tokens=False, padding=False,
                      truncation=True, max_length=max_target_len)

    pad_id = tokenizer.pad_token_id
    seqs: List[Tuple[List[int], int]] = []
    max_len = 0
    for pp, rr in zip(p_ids["input_ids"], r_ids["input_ids"]):
        ids = pp + rr
        seqs.append((ids, len(pp)))
        max_len = max(max_len, len(ids))

    input_ids, attn_masks, labels, resp_masks = [], [], [], []
    for ids, p_len in seqs:
        pad_len = max_len - len(ids)
        x  = ids + [pad_id] * pad_len
        am = [1]*len(ids) + [0]*pad_len
        lab = x.copy()
        for i in range(p_len): lab[i] = -100       # ignore prompt in loss
        rm = [0]*p_len + [1]*(len(ids)-p_len) + [0]*pad_len  # mask only reply tokens
        input_ids.append(x); attn_masks.append(am); labels.append(lab); resp_masks.append(rm)

    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn_masks, dtype=torch.long),
            torch.tensor(labels,     dtype=torch.long),
            torch.tensor(resp_masks, dtype=torch.bool))

def mean_logprobs(model, input_ids, attention_mask, labels, resp_mask,
                  dtype, autocast_enabled=True):
    # Return length-normalized sequence logprob and token counts
    if torch.cuda.is_available() and autocast_enabled:
        ctx = autocast(device_type="cuda", dtype=dtype)
    else:
        class _NoOp:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        ctx = _NoOp()
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask   = resp_mask[:, 1:]

    valid = (shift_labels != -100) & shift_mask
    logp = torch.log_softmax(shift_logits, dim=-1)
    gather_idx = shift_labels.masked_fill(~valid, 0).unsqueeze(-1)
    tok_logp = torch.gather(logp, -1, gather_idx).squeeze(-1)
    tok_logp = tok_logp * valid.float()

    token_counts = valid.float().sum(dim=1).clamp_min(1.0)
    seq_logp_mean = tok_logp.sum(dim=1) / token_counts
    return seq_logp_mean, token_counts

# ---------------- Simple CSV logger ----------------
class CSVLogger:
    def __init__(self, path): self.path=path; self._init=False
    def log(self, row:dict):
        if not self.path: return
        d = os.path.dirname(os.path.abspath(self.path))
        if d: os.makedirs(d, exist_ok=True)
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        write_header = not os.path.exists(self.path) or not self._init
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)
        self._init=True

# ---------------- Main training ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_model_dir", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    # lengths
    ap.add_argument("--max_prompt_length",     type=int, default=1024)
    ap.add_argument("--max_target_length",     type=int, default=256)
    ap.add_argument("--max_pa_prompt_length",  type=int, default=768)

    # loss hyperparams
    ap.add_argument("--beta",      type=float, default=0.05)
    ap.add_argument("--lambda_pa", type=float, default=1.0)

    # training hyperparams
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    # precision / device
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # LoRA
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", nargs="*", default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

    # logging
    ap.add_argument("--csv_log", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_logger = CSVLogger(args.csv_log) if args.csv_log else None

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    amp_dtype = (torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)) if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None
    print(f"[DEVICE] CUDA={'yes' if use_cuda else 'no'} | device_map={device_map} | dtype={amp_dtype}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir, use_fast=False, trust_remote_code=True)
    tokenizer = add_special_tokens(tokenizer)

    # Models
    policy = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir, device_map=device_map, trust_remote_code=True,
        torch_dtype=amp_dtype if use_cuda else torch.float32
    )
    ref = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir, device_map=device_map, trust_remote_code=True,
        torch_dtype=amp_dtype if use_cuda else torch.float32
    )
    policy.resize_token_embeddings(len(tokenizer))
    ref.resize_token_embeddings(len(tokenizer))
    policy.config.use_cache = False
    ref.config.use_cache = False

    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()

    # Optional LoRA for policy
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets
        )
        policy = get_peft_model(policy, lora)
        try:
            policy.print_trainable_parameters()
        except Exception:
            pass

    ref.requires_grad_(False); ref.eval(); policy.train()

    # Data
    train_ds = PADPODataset(args.train_jsonl)
    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size,
                              shuffle=True, drop_last=False, pin_memory=use_cuda)
    total_steps = math.ceil(len(train_loader) * args.num_train_epochs / max(1, args.gradient_accumulation_steps))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    # Opt/sched
    optim = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    print(f"[INFO] Train samples: {len(train_ds)} | Batches/epoch: {len(train_loader)} | Total updates: {total_steps}")
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in policy.parameters())
    print(f"[PARAMS] Trainable: {trainable:,} / {total:,}")

    global_step, accum = 0, 0

    def do_tokenize(x_list, y_list, max_p_len, max_t_len):
        return concat_and_tokenize(tokenizer, x_list, y_list, max_p_len, max_t_len)

    for epoch in range(int(math.ceil(args.num_train_epochs))):
        for it, batch in enumerate(train_loader, start=1):
            x        = batch["x"]
            y_plus   = [wrap_reply(t) for t in batch["y_plus"]]
            y_minus  = [wrap_reply(t) for t in batch["y_minus"]]
            pa_plus  = [_norm_pa(p) for p in (batch["pa_plus"]  if isinstance(batch["pa_plus"],  list) else [batch["pa_plus"]])]
            pa_minus = [_norm_pa(p) for p in (batch["pa_minus"] if isinstance(batch["pa_minus"], list) else [batch["pa_minus"]])]
            if len(pa_plus)  == 1 and len(x) > 1:  pa_plus  = pa_plus  * len(x)
            if len(pa_minus) == 1 and len(x) > 1:  pa_minus = pa_minus * len(x)

            prompt_x = [xx for xx in x]
            prompt_x_pa_plus  = [xx + "\n\n" + serialize_pa(pa["persona"], pa["act"])  for xx, pa in zip(x, pa_plus)]
            prompt_x_pa_minus = [xx + "\n\n" + serialize_pa(pa["persona"], pa["act"])  for xx, pa in zip(x, pa_minus)]

            # tokenize
            A_inp,A_att,A_lab,A_mask = do_tokenize(prompt_x,          y_plus,  args.max_prompt_length,    args.max_target_length)
            B_inp,B_att,B_lab,B_mask = do_tokenize(prompt_x,          y_minus, args.max_prompt_length,    args.max_target_length)
            C_inp,C_att,C_lab,C_mask = do_tokenize(prompt_x_pa_plus,  y_plus,  args.max_pa_prompt_length, args.max_target_length)
            D_inp,D_att,D_lab,D_mask = do_tokenize(prompt_x_pa_minus, y_minus, args.max_pa_prompt_length, args.max_target_length)

            dev = next(policy.parameters()).device
            A_inp=A_inp.to(dev); A_att=A_att.to(dev); A_lab=A_lab.to(dev); A_mask=A_mask.to(dev)
            B_inp=B_inp.to(dev); B_att=B_att.to(dev); B_lab=B_lab.to(dev); B_mask=B_mask.to(dev)
            C_inp=C_inp.to(dev); C_att=C_att.to(dev); C_lab=C_lab.to(dev); C_mask=C_mask.to(dev)
            D_inp=D_inp.to(dev); D_att=D_att.to(dev); D_lab=D_lab.to(dev); D_mask=D_mask.to(dev)

            # logprobs (means) + token counts
            lp_pos_pi,  npos  = mean_logprobs(policy, A_inp, A_att, A_lab, A_mask, dtype=amp_dtype, autocast_enabled=use_cuda)
            lp_neg_pi,  nneg  = mean_logprobs(policy, B_inp, B_att, B_lab, B_mask, dtype=amp_dtype, autocast_enabled=use_cuda)
            lpa_pos_pi, npos2 = mean_logprobs(policy, C_inp, C_att, C_lab, C_mask, dtype=amp_dtype, autocast_enabled=use_cuda)
            lpa_neg_pi, nneg2 = mean_logprobs(policy, D_inp, D_att, D_lab, D_mask, dtype=amp_dtype, autocast_enabled=use_cuda)

            with torch.no_grad():
                lp_pos_ref,  _ = mean_logprobs(ref, A_inp, A_att, A_lab, A_mask, dtype=torch.float32, autocast_enabled=False)
                lp_neg_ref,  _ = mean_logprobs(ref, B_inp, B_att, B_lab, B_mask, dtype=torch.float32, autocast_enabled=False)
                lpa_pos_ref, _ = mean_logprobs(ref, C_inp, C_att, C_lab, C_mask, dtype=torch.float32, autocast_enabled=False)
                lpa_neg_ref, _ = mean_logprobs(ref, D_inp, D_att, D_lab, D_mask, dtype=torch.float32, autocast_enabled=False)

            # advantages in fp32 + clamp
            beta = float(args.beta)
            dpo_adv  = ((lp_pos_pi - lp_neg_pi) - (lp_pos_ref - lp_neg_ref)).to(torch.float32)
            pa_adv   = ((lpa_pos_pi - lpa_neg_pi) - (lpa_pos_ref - lpa_neg_ref)).to(torch.float32)
            z_dpo    = (beta * dpo_adv).clamp_(-20.0, 20.0)
            z_pa     = (beta * pa_adv ).clamp_(-20.0, 20.0)

            loss_dpo = -torch.log(torch.sigmoid(z_dpo) + 1e-12).mean()
            loss_pa  = -torch.log(torch.sigmoid(z_pa ) + 1e-12).mean()
            loss     = loss_dpo + args.lambda_pa * loss_pa

            if not torch.isfinite(loss):
                print("[WARN] non-finite loss; skipping step")
                optim.zero_grad(set_to_none=True); continue

            (loss / max(1, args.gradient_accumulation_steps)).backward()
            accum += 1

            if (accum % args.gradient_accumulation_steps) == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
                global_step += 1

                if (global_step <= 3) or (global_step % args.log_every == 0) or (global_step == total_steps):
                    lr = sched.get_last_lr()[0]
                    with torch.no_grad():
                        mean_tokens = 0.25*(npos.mean()+nneg.mean()+npos2.mean()+nneg2.mean()).item()
                        dbg_adv_dpo = dpo_adv.mean().item()
                        dbg_adv_pa  = pa_adv.mean().item()
                    msg = (f"step {global_step}/{total_steps} | "
                           f"loss {loss.item():.4f} | dpo {loss_dpo.item():.4f} | pa {loss_pa.item():.4f} | "
                           f"adv_dpo {dbg_adv_dpo:+.3f} | adv_pa {dbg_adv_pa:+.3f} | "
                           f"tok/reply ~{mean_tokens:.1f} | lr {lr:.2e}")
                    print(msg)
                    if csv_logger:
                        csv_logger.log({
                            "phase":"train","step":global_step,
                            "loss":float(loss.item()),"loss_dpo":float(loss_dpo.item()),"loss_pa":float(loss_pa.item()),
                            "adv_dpo":float(dbg_adv_dpo),"adv_pa":float(dbg_adv_pa),
                            "tok_per_reply":float(mean_tokens),"lr":float(lr)
                        })

                if (global_step % args.save_every == 0) or (global_step == total_steps):
                    sd = os.path.join(args.output_dir, f"step_{global_step}")
                    os.makedirs(sd, exist_ok=True)
                    # save policy only
                    policy.save_pretrained(sd); tokenizer.save_pretrained(sd)

        # flush partial accumulation if any
        if (accum % args.gradient_accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)

        ed = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(ed, exist_ok=True)
        policy.save_pretrained(ed); tokenizer.save_pretrained(ed)

    policy.save_pretrained(args.output_dir); tokenizer.save_pretrained(args.output_dir)
    print("✅ Training complete. Saved to", args.output_dir)

if __name__ == "__main__":
    main()

    
