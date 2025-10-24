import os
import pandas as pd
from openai import OpenAI
import numpy as np

client = OpenAI(api_key="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")
import re
from tqdm import tqdm

# Set your OpenAI API key

# Load dataset
df = pd.read_csv("ABN_final_dataset.csv")

# Persona profile definitions
BUYER_PROFILES = {
    0: "Quality concerned prioritizes high standards in amenities and services, seeking assurance of superior quality regardless of cost.",
    1: "Budget Concerned seeks cost-effective options, emphasizing value for money and actively comparing prices to maximize their budget.",
    2: "Quality and Budget concerned Traveler balances quality and cost, seeking well-reviewed options that meet both quality standards and budget constraints."
}

ARGUMENTATION_PROFILES = {
    0: "Agreeable: Accepts offers and arguments with minimal conflict. Prioritizes mutual agreement, shows flexibility.",
    1: "Disagreeable: Rejects proposals unless compelling. Frequently challenges positions to satisfy self-interest.",
    2: "Open-minded: Evaluates offers rationally, neither blindly accepts nor rejects. Willing to explore alternatives.",
    3: "Argumentative: Strongly challenges opposing offers, debates actively, seeks to dominate through counterarguments."
}

# Prompt templates
COT_PROMPT_TEMPLATE = """
**Role:** You are an expert psychological analyst specializing in negotiation styles.

**Context:**
A Traveler with the following known persona profiles is in a negotiation dialogue with a Travel Agent.
- **Argumentation Profile:** {arg_prof}
- **Preference Profile:** {pref_prof}
- **Buying Style Profile:** {buyer_prof}

Here is the dialogue transcript:
---
{dialogue}
---

**Your Task:**
Analyze the **Traveler's** statements ONLY. Your goal is to extract persona triplets that are consistent with the Traveler's known profiles. Perform the following two steps for every piece of supporting evidence you find:

1. **Identify Evidence:** Quote the exact sentence(s) from the Traveler that supports one of their given persona profiles.
2. **Extract Triplet:** Based ONLY on the evidence you quoted, generate a single, structured persona triplet in the format (Subject, Predicate, Object) that exemplifies the corresponding profile. The Subject should be "Traveler".

Avoid generic predicates like "is", "has", "shows". Be specific, e.g., "prefers", "seeks", "argues for", "negotiates", "prioritizes".

Output format:
**Utterance:** <original utterance>
**Evidence:** <exact quote>
**Triplet:** (Traveler, predicate, object)
"""

VALIDATION_PROMPT_TEMPLATE = """
**Role:** You are a meticulous data validation expert. Your task is to ensure perfect alignment between evidence, a given persona profile, and the extracted conclusion.

**Context:**
A Traveler with the following known persona profiles was in a negotiation dialogue.
- **Argumentation Profile:** {arg_prof}
- **Preference Profile:** {pref_prof}
- **Buying Style Profile:** {buyer_prof}

Here is the dialogue transcript:
---
{dialogue}
---
Here is a list of persona triplets that were extracted to match the profiles:
---
{triplets}
---

**Your Task:**
Review each triplet. Your goal is to return a final, clean list containing ONLY the triplets that are a **strong and direct inference** from the Traveler's statements and **accurately reflect the given persona profiles**.

Return the validated list in the same format:
**Utterance:** <original utterance>
**Evidence:** <exact quote>
**Triplet:** (Traveler, predicate, object)
"""

# Helper functions
def generate_dialogue_text(sub_df):
    return "\n".join([
        f"{'Agent' if row['Speaker'] == 1 else 'Traveler'}: {row['Utterance']}"
        for _, row in sub_df.iterrows()
    ])

def extract_validated_triplets(text):
    pattern = r"\*\*Utterance:\*\* (.*?)\n\*\*Evidence:\*\* (.*?)\n\*\*Triplet:\*\* \((Traveler),\s*([^,\n]+?),\s*([^\n\)]+?)\)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(utt.strip(), ev.strip(), f"({s.strip()}, {p.strip()}, {o.strip()})") for utt, ev, s, p, o in matches]

# Add new columns
df["persona_evidences"] = pd.NA
df["persona_triplets"] = pd.NA

# Group by conv_id
conversations = df.groupby("conv_id")

# Deterministically select first 10% of conv_ids (sorted)
all_conv_ids = sorted(conversations.groups.keys())
cutoff = int(len(all_conv_ids) * 0.50)
selected_conv_ids = all_conv_ids[:cutoff]

print(f"⚡ Processing deterministically: first {cutoff} out of {len(all_conv_ids)} conv_ids (10%)")

for i, conv_id in enumerate(selected_conv_ids):
    conv_df = conversations.get_group(conv_id)
    print(f"\n🔍 Processing conversation {i+1} → conv_id = {conv_id}")

    traveler_rows = conv_df[conv_df["Speaker"] == 0]
    if traveler_rows.empty:
        continue

    first_row = traveler_rows.iloc[0]
    arg_profile = ARGUMENTATION_PROFILES.get(first_row["Buyer Argument Profile"], "Unknown")
    pref_profile = first_row["Preference Profile"]
    buyer_profile = BUYER_PROFILES.get(first_row["Buyer Profile"], "Unknown")

    dialogue = generate_dialogue_text(conv_df)

    # Step 1: CoT Prompt
    cot_prompt = COT_PROMPT_TEMPLATE.format(
        arg_prof=arg_profile,
        pref_prof=pref_profile,
        buyer_prof=buyer_profile,
        dialogue=dialogue
    )
    cot_response = client.chat.completions.create(model="gpt-4.1-nano-2025-04-14",
    messages=[{"role": "user", "content": cot_prompt}],
    temperature=0.5)
    raw_triplets = cot_response.choices[0].message.content.strip()

    # Step 2: Validation
    validation_prompt = VALIDATION_PROMPT_TEMPLATE.format(
        arg_prof=arg_profile,
        pref_prof=pref_profile,
        buyer_prof=buyer_profile,
        dialogue=dialogue,
        triplets=raw_triplets
    )
    validation_response = client.chat.completions.create(model="gpt-4.1-nano-2025-04-14",
    messages=[{"role": "user", "content": validation_prompt}],
    temperature=0.2)
    validated_text = validation_response.choices[0].message.content.strip()

    # Step 3: Extract final validated triplets
    validated = extract_validated_triplets(validated_text)
    if not validated:
        continue

    evidences = [ev for _, ev, _ in validated]
    triplets = [trip for _, _, trip in validated]

    first_idx = conv_df.index[0]
    df.at[first_idx, "persona_evidences"] = evidences
    df.at[first_idx, "persona_triplets"] = triplets

# Save final CSV
df.to_csv("ABN_final_dataset_with_triplets_by_conv_50.csv", index=False)
print("\n✅ Triplet summary added to first row of each conversation. File saved.")
