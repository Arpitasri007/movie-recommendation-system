import pandas as pd

df = pd.read_csv("indian movies.csv")

result = df[
    df["Movie Name"].str.contains(
        "haq",
        case=False,
        na=False
    )
]

print(result[["Movie Name", "Language"]].head(50))