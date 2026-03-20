from datasets import load_dataset
import pandas as pd

# ── AI sources ───────────────────────────────────────────

# 1. Student essays (ChatGPT written) — andythetechnerd03
essays = load_dataset("andythetechnerd03/AI-human-text", split="train")
essays_df = essays.to_pandas()
essays_ai = pd.DataFrame({
    "content_text": essays_df[essays_df["generated"]==1]["text"].dropna(),
    "author_type": "AI"
})
essays_human = pd.DataFrame({
    "content_text": essays_df[essays_df["generated"]==0]["text"].dropna(),
    "author_type": "Human"
})

# 2. HC3 ChatGPT answers — QA style AI
hc3 = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
hc3_df = hc3.to_pandas()
hc3_ai = pd.DataFrame({
    "content_text": hc3_df["chatgpt_answers"].explode().dropna(),
    "author_type": "AI"
})
hc3_human = pd.DataFrame({
    "content_text": hc3_df["human_answers"].explode().dropna(),
    "author_type": "Human"
})

# 3. GPT-2 Wikipedia intros — encyclopedia style AI
gpt2 = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
gpt2_df = gpt2.to_pandas()
gpt2_ai = pd.DataFrame({
    "content_text": gpt2_df["generated_intro"].dropna(),
    "author_type": "AI"
})
# Real Wikipedia intros as human counterpart
wiki_human = pd.DataFrame({
    "content_text": gpt2_df["wiki_intro"].dropna(),
    "author_type": "Human"
})

# ── Combine ──────────────────────────────────────────────
all_ai = pd.concat([essays_ai, hc3_ai, gpt2_ai]).dropna().reset_index(drop=True)
all_ai = all_ai[all_ai["content_text"].str.len() > 100]

all_human = pd.concat([essays_human, hc3_human, wiki_human]).dropna().reset_index(drop=True)
all_human = all_human[all_human["content_text"].str.len() > 100]

print(f"AI pool: {len(all_ai)}, Human pool: {len(all_human)}")

n = min(len(all_ai), len(all_human), 30000)
final_df = pd.concat([
    all_ai.sample(n, random_state=42),
    all_human.sample(n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

final_df["content_text"] = final_df["content_text"].str.strip()
final_df = final_df[final_df["content_text"].str.len() > 100].reset_index(drop=True)
final_df.to_csv("data.csv", index=False)