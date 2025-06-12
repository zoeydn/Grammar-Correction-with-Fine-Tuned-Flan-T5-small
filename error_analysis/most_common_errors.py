import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

txt_path = "full_error_analysis.txt"

with open(txt_path, "r") as f:
    lines = f.readlines()

data_str = "".join(lines)
df = pd.read_fwf(StringIO(data_str))
df.columns = df.columns.str.strip()

df["TotalIncorrect"] = df["FalsePositive"] + df["FalseNegative"]

top_wrong = df.sort_values(by="TotalIncorrect", ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.bar(top_wrong["EditType"], top_wrong["FalseNegative"], label="Missed (FN)", alpha=0.8)
plt.bar(top_wrong["EditType"], top_wrong["FalsePositive"], bottom=top_wrong["FalseNegative"], label="Wrong Edit (FP)", alpha=0.6)
plt.ylabel("Error Count")
plt.title("Top 10 Grammar Errors the Model Failed to Handle (Missed or Incorrectly Fixed)")
plt.legend()
plt.tight_layout()

plt.savefig("full_model_failed_or_wrong_fixes.png", dpi=300)
plt.show()
