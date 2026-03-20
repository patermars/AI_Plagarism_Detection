from datasets import load_dataset
import pandas as pd

ds = load_dataset("Hello-SimpleAI/HC3", "all")
df = ds["train"].to_pandas()

df = df[["human_answers", "chatgpt_answers"]].explode("human_answers").explode("chatgpt_answers")

human_df = pd.DataFrame({"content_text": df["human_answers"].dropna(), "author_type": "Human"})
ai_df    = pd.DataFrame({"content_text": df["chatgpt_answers"].dropna(), "author_type": "AI"})

final_df = pd.concat([human_df, ai_df]).dropna().sample(frac=1, random_state=42).reset_index(drop=True)
final_df.to_csv("data.csv", index=False)
print(final_df["author_type"].value_counts())