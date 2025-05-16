# src/ner_model.py

import torch
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
import pickle
from torch.nn import LayerNorm
import os

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % self.num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, values)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out(context), attn_weights


class BiLSTM_CRF_Attn(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, embedding_matrix=None, dropout=0.5, num_heads=4):
        super(BiLSTM_CRF_Attn, self).__init__()

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attn_layer = MultiHeadAttentionLayer(hidden_dim * 2, num_heads=num_heads)
        self.layer_norm = LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        attn_out, attn_weights = self.attn_layer(lstm_out)

        lstm_out = self.dropout(self.layer_norm(lstm_out))
        attn_out = self.dropout(self.layer_norm(attn_out))

        emissions = self.fc(attn_out)
        return emissions, attn_out, attn_weights

    def predict(self, x):
        emissions, _, _ = self.forward(x)
        return self.crf.decode(emissions)

## NER Class

class NEREngine:
    def __init__(self, config_path="model/ner_config.pkl", model_path="model/ner_model.pth"):
        # Load model config: vocab_size, embed_dim, hidden_dim, num_tags, etc.
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError("NER model or config not found.")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        self.word_to_index = config["word_to_index"]
        self.tag_to_index = config["tag_to_index"]
        self.index_to_tag = {v: k for k, v in self.tag_to_index.items()}

        self.model = BiLSTM_CRF_Attn(
            vocab_size=len(self.word_to_index),
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_tags=len(self.tag_to_index),
            embedding_matrix=config.get("embedding_matrix", None)
        )

        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def preprocess(self, tokens):
        # Convert tokens to indices using word_to_index (fallback to unknown token)
        unk_idx = self.word_to_index.get("__UNK__", 0)
        indices = [self.word_to_index.get(token, unk_idx) for token in tokens]
        return torch.tensor([indices], dtype=torch.long)

    def predict_tags(self, tokens):
        x = self.preprocess(tokens)
        with torch.no_grad():
            tag_ids = self.model.predict(x)[0]  # Output is List[List[int]]
        return [self.index_to_tag[i] for i in tag_ids]
