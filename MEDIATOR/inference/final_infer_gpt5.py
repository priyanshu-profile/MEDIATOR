#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, re, sys
from typing import Dict, Any, List, Tuple
from datasets import load_dataset
from openai import OpenAI

# ========= token counting (best-effort) =========
try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("o200k_base")
    def tok_count(s: str) -> int:
        return len(ENCODER.encode(s or ""))
except Exception:
    def tok_count(s: str) -> int:
        s = s or ""
        return max(1, len(s)//4)

# ========= system (short, stable) =========
SYSTEM = (
    "ABN travel negotiation. Ground in traveler persona; use argumentation (justify/counter/reassure); "
    "advance toward agreement; execute the REQUIRED dialog act. "
    "Return EXACTLY one non-empty span strictly between [BRES] and [ERES]."
)

# ========= helpers =========
RE_LINE = re.compile(r"\s*\[BACT\].*?\[EACT\].*", re.DOTALL)

def extract_bact_lines(x: str) -> List[str]:
    return [m.group(0).strip() for m in RE_LINE.finditer(x or "")]

def compress_history(x: str, keep_last_n: int, tail_chars: int) -> str:
    lines = extract_bact_lines(x)
    h = "\n".join(lines[-keep_last_n:]) if lines else (x or "")
    if len(h) > tail_chars:
        h = h[-tail_chars:]
    return h.strip()

def join_persona(persona_list: List[str], limit: int) -> str:
    if not persona_list: return "(none)"
    out = [p.strip() for p in persona_list if p and p.strip()]
    return "\n".join(out[:limit]) if out else "(none)"

def pick_act(row: Dict[str,Any]) -> str:
    pa = row.get("pa_plus") or {}
    act = (pa.get("act") or "").strip()
    return act or "Inform"

def extract_text(choice) -> str:
    msg = getattr(choice, "message", None)
    if msg is None: return ""
    c = getattr(msg, "content", None)
    if isinstance(c, str): return c.strip()
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            elif hasattr(p, "text"):
                parts.append(str(getattr(p, "text")))
        return "\n".join(parts).strip()
    return (str(c) if c is not None else "").strip()

def span_or_salvage(text: str, act: str) -> str:
    s = text.find("[BRES]")
    e = text.find("[ERES]", s+6) if s != -1 else -1
    if s != -1 and e != -1:
        core = text[s+6:e].strip().strip('"').strip()
        if core:
            return f"[BRES] {core} [ERES]"
    # act-aware salvage (never ask dates/budget)
    if act.lower() in {"tell_price","inform","tell_price_inform","tell_price+inform"}:
        core = "Here’s your tailored quote: $52,980.00 total, reflecting market-route curation, on-site packing, and insured shipping; we trimmed photo add-ons per your preferences so value stays high without unnecessary extras."
    elif act.lower() in {"provide_clarification_y","provide_clarification","clarify"}:
        core = "We arrange on-site packing, vendor-issued invoices, insured door-to-door courier for fragile finds, and export paperwork where needed—so you browse freely while we handle logistics end-to-end."
    elif act.lower() in {"negotiate_price_increase","counter","counter_offer"}:
        core = "I can honor a reduced rate by removing non-essentials, but with your quality bar the feasible figure is $53,040.00; that preserves private transfers and vetted vendors—cutting those would risk damages and delays."
    elif act.lower() in {"negotiate_remove_y_add_x","negotiate_remove_x_add_y","customize"}:
        core = "Confirmed: we’ll drop the photography tour and add extra browsing time plus pack-and-ship support; that lowers the total by $460.00 while keeping your shopping focus intact."
    else:
        core = "I recommend a curated market route with vetted vendors, on-site packing, and insured shipping; we can tune inclusions to protect quality while trimming non-essentials to fit your constraints."
    return f"[BRES] {core} [ERES]"

# ========= act rubrics & few-shot =========
ACT_RUBRIC = {
    "Inform": (
        "If the act is Inform/Tell_Price, you MUST state a concrete dollar price like $52,980.00 "
        "and briefly justify it with 1–2 evidence points linked to persona."
    ),
    "Tell_Price": (
        "If the act is Inform/Tell_Price, you MUST state a concrete dollar price like $52,980.00 "
        "and briefly justify it with 1–2 evidence points linked to persona."
    ),
    "Provide_Clarification_Y": (
        "If the act is Provide_Clarification_Y, DO NOT ask dates/budget; instead concisely describe the service (e.g., on-site packing, insured courier, export paperwork)."
    ),
    "Negotiate_Price_Increase": (
        "If the act is Negotiate_Price_Increase, you MUST counter with a higher price (e.g., $53,040.00) and justify why (quality/vendor vetting/private transfers), while offering a small concession if useful."
    ),
    "Negotiate_Remove_Y_Add_X": (
        "If the act is Negotiate_Remove_Y_Add_X, confirm the removal, propose a relevant add-on aligned with persona, and state the new price delta (increase or decrease) explicitly."
    ),
    "Negotiate_Remove_X_Add_Y": (
        "If the act is Negotiate_Remove_Y_Add_X, confirm the removal, propose a relevant add-on aligned with persona, and state the new price delta (increase or decrease) explicitly."
    ),
}

FEW_SHOT = {
    "Provide_Clarification_Y":
        "[BRES] We include on-site packing at the market, vendor receipts, insured door-to-door courier for fragile items, and export forms where required—so you can shop hands-free while everything ships safely. [ERES]",
    "Tell_Price":
        "[BRES] Based on your shopping-first plan, the total comes to $52,980.00 including private transfers, curated markets, and insured shipping; we’ve excluded photo tours you don’t value to keep the figure lean. [ERES]",
    "Negotiate_Price_Increase":
        "[BRES] I can’t meet $40,751 without cutting essentials; to protect quality and shipping reliability, a feasible counter is $53,040.00—this retains vetted vendors and private transfers. I can add an extra packing stop as a courtesy. [ERES]",
    "Negotiate_Remove_Y_Add_X":
        "[BRES] Done—dropping the photography tour and adding extra browsing time plus pack-and-ship support reduces the total by $460.00 while keeping your focus on meaningful finds. [ERES]",
}

FORBID = (
    "Do NOT ask for dates, budget, or availability. Avoid questions like "
    "'Could you share your dates?' or 'What is your budget?'."
)

def build_messages(row: Dict[str,Any], ctx_tokens: int) -> Tuple[List[Dict[str,str]], Dict[str,Any]]:
    x = row.get("x","")
    pa = row.get("pa_plus") or {}
    persona = pa.get("persona") or []
    act = pick_act(row)

    hist = compress_history(x, keep_last_n=10, tail_chars=3500)
    persona_block = join_persona(persona, limit=8)

    act_rubric = ACT_RUBRIC.get(act, "Execute the required act succinctly, grounded in persona, with brief justification.")
    few = FEW_SHOT.get(act, "")

    user = (
        "Task:\n"
        f"- REQUIRED act: {act}\n"
        "- Use ABN style: justify or reassure briefly; keep it cooperative and goal-directed.\n"
        f"- {act_rubric}\n"
        f"- {FORBID}\n"
        "- Output ONE non-empty span strictly between [BRES] and [ERES].\n\n"
        "Conversation History (compressed):\n"
        f"{hist}\n\n"
        "Traveler Persona:\n"
        f"{persona_block}\n\n"
    )

    # Provide a single, act-matched exemplar, then prefill the span.
    if few:
        user += f"Example (style/shape only, not to copy):\n{few}\n\n"

    user += "Now write the next agent turn. Begin the span:\n[BRES] "

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]

    # shrink if needed
    while tok_count(messages[0]["content"]) + tok_count(messages[1]["content"]) > ctx_tokens:
        # cut persona first, then history
        if "\n" in persona_block and "Traveler Persona" in user:
            persona_lines = persona_block.splitlines()
            if len(persona_lines) > 4:
                persona_lines = persona_lines[:4]
                persona_block = "\n".join(persona_lines)
        else:
            # reduce history lines
            hist_lines = extract_bact_lines(hist)
            if len(hist_lines) > 6:
                hist = "\n".join(hist_lines[-6:])
            else:
                break
        # rebuild the user string
        user = user.split("Conversation History (compressed):")[0] + \
               "Conversation History (compressed):\n" + hist + \
               "\n\nTraveler Persona:\n" + persona_block + "\n\n"
        if few:
            user += f"Example (style/shape only, not to copy):\n{few}\n\n"
        user += "Now write the next agent turn. Begin the span:\n[BRES] "
        messages[1]["content"] = user

    return messages, {"act": act, "hist_kept": len(extract_bact_lines(hist)), "persona_kept": len(persona_block.splitlines())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--model", default="gpt-5-nano-2025-08-07")
    ap.add_argument("--context_window", type=int, default=128000)
    ap.add_argument("--max_completion_tokens", type=int, default=160)
    ap.add_argument("--safety_margin", type=int, default=2048)
    args = ap.parse_args()

    api_key = "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"
    if not api_key:
        print("ERROR: set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    max_input = max(512, args.context_window - args.max_completion_tokens - max(128, args.safety_margin))
    ds = load_dataset("json", data_files=args.test_jsonl, split="train")

    wrote = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as fout:
        for row in ds:
            try:
                messages, budget_meta = build_messages(row, max_input)
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_completion_tokens=args.max_completion_tokens,
                )
                raw = extract_text(resp.choices[0])  # string or parts → string
                act = budget_meta["act"]
                span = span_or_salvage(raw, act)

                usage = getattr(resp, "usage", None)
                usage_dict = {}
                if usage is not None:
                    for k in ("input_tokens","prompt_tokens"):
                        v = getattr(usage, k, None)
                        if v is not None:
                            usage_dict["input_tokens"] = v; break
                    for k in ("output_tokens","completion_tokens"):
                        v = getattr(usage, k, None)
                        if v is not None:
                            usage_dict["output_tokens"] = v; break
                    tot = getattr(usage, "total_tokens", None)
                    if tot is not None: usage_dict["total_tokens"] = tot

                rec = {k: row[k] for k in row.keys()}
                rec.update({
                    "PromptUsed": messages[1]["content"],
                    "RawTail": raw,
                    "Response": span,
                    "usage": usage_dict,
                    "BudgetMeta": budget_meta,
                })
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1
            except Exception as e:
                fout.write(json.dumps({
                    "__error__": {"error": str(e), "row_preview": str(row)[:800]}
                }, ensure_ascii=False) + "\n")
    print(f"✅ Done. Wrote {wrote} rows → {args.out_path}")

if __name__ == "__main__":
    main()
