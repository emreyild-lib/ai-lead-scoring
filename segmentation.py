import pandas as pd

df = pd.read_csv("scored_leads.csv")

# Rank leads by score
df = df.sort_values("lead_score", ascending=False).reset_index(drop=True)

# Assign segment based on quantiles
df["segment"] = pd.qcut(df["lead_score"], q=3, labels=["Cold", "Warm", "Hot"])

df.to_csv("scored_leads_segmented.csv", index=False)
print(df.head(10))
