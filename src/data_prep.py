import argparse
import pandas as pd

# Get input and output file paths

parser = argparse.ArgumentParser()
parser.add_argument("--df1", required=True)
parser.add_argument("--df2", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

# load data in pandas dfs

oas_df = pd.read_csv(args.df1)
thera_df = pd.read_csv(args.df2)

# get sequences and add classes for downstream classification

oas_clean = oas_df[['sequence']]
oas_clean.insert(1, 'class', 0)

thera_clean = thera_df[['HeavySequence']]
thera_clean = thera_clean.rename(columns={'HeavySequence': 'sequence'})
thera_clean.insert(1, 'class', 1)

# Find shortest df
min_rows = min(len(oas_clean), len(thera_clean))

# match dfs with random sampling


def match_df(df, n):

    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    return df


oas_clean = match_df(oas_clean, min_rows)
thera_clean = match_df(thera_clean, min_rows)

frames = [oas_clean, thera_clean]

# join and randomize dfs
output_df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=343)

output_df.to_csv(args.out, index=False)
