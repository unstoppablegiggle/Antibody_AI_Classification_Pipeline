import argparse
import pandas as pd
from transformers import AutoModel, AutoTokenizer
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

# establish model
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D", trust_remote_code=True,
                                  cache_dir="./models")
model = model.to(device)

# get sequences


def encode_seqs(seqs, tokenizer, model, batch_size=32):

    encoded_seqs = []
    model_outputs = []
    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i:i+batch_size]

        encoded_input = tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():  # don't need gradients
            model_output = model(**encoded_input)

        for j in range(len(batch_seqs)):
            seq_encoding = {k: v[j:j+1] for k, v in encoded_input.items()}
            seq_model_output = {'last_hidden_state': model_output.last_hidden_state[j:j + 1]}
            encoded_seqs.append(seq_encoding)
            model_outputs.append(seq_model_output)

    return encoded_seqs, model_outputs


def get_sequence_embeddings(encoded_seqs, model_outputs):

    mean_embeds = []

    for s, m in zip(encoded_seqs, model_outputs):
        mask = s['attention_mask'].float()
        mask[:, 0] = 0.0  # make cl tokens invisible
        mask = mask.unsqueeze(-1).expand(m['last_hidden_state'].size())

        sum_embeddings = torch.sum(m['last_hidden_state'] * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        mean_embeds.append(mean_pooled.squeeze(0).cpu().tolist())

    return mean_embeds


encoded_seqs, model_output = encode_seqs(seqs, tokenizer, model)

mean_embeddings = get_sequence_embeddings(encoded_seqs, model_output)

embed_names = [f"em_{i:03d}" for i in range(len(mean_embeddings[0]))]

embeddings_df = pd.DataFrame(mean_embeddings, columns=embed_names)

output_df = df.copy().join(embeddings_df)
output_df['embeds'] = mean_embeddings

output_df.to_csv(args.out, index=False)
