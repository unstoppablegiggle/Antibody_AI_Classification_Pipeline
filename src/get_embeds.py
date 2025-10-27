
import argparse
import pandas as pd
import ablang2
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get input and output file paths

parser = argparse.ArgumentParser()
parser.add_argument("--df", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

df = pd.read_csv(args.df)

# pull seqs for processing
seqs = df['sequence'].tolist()
paired_seqs = [[seq.upper(), ''] for seq in seqs]  # heavy chain only processing

# establish model
ablang = ablang2.pretrained(
    model_to_use='ablang2-paired',
    random_init=False,
    ncpu=1,
    device=device
)

# get embeddings
embeddings = ablang(paired_seqs, mode='seqcoding')

embed_names = [f"em_{i:03d}" for i in range(len(embeddings[0]))]

embeddings_df = pd.DataFrame(embeddings, columns=embed_names)

# prepare output df
output_df = df.copy().join(embeddings_df)

output_df.to_csv(args.out, index=False)
