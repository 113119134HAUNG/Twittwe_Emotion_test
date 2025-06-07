# model.py

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.norm(x + attn_output)


class ResidualClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(self.norm1(x))
        x = self.dropout(x)

        residual = x
        x = self.fc2(x)
        x = self.activation(self.norm2(x + residual))
        x = self.dropout(x)

        return self.out(x)


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=128,
                 num_heads=4, num_classes=3, emotion_dim=None):
        super().__init__()
        assert emotion_dim is not None
        self.emotion_dim = emotion_dim
        self.embedding_dim = embedding_dim
        self.input_dim = embedding_dim + emotion_dim

        # === Embedding ===
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.3)

        # === Emotion Gating Layer ===
        self.emotion_gate = nn.Sequential(
            nn.Linear(emotion_dim, 1),
            nn.Sigmoid()  # Gate: [B, L, 1]
        )

        # === LSTM Encoder ===
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim * 2)

        # === Attention Layer ===
        self.attention = MultiHeadAttention(embed_dim=hidden_dim * 2, num_heads=num_heads)

        # === Residual Classifier ===
        self.classifier = ResidualClassifier(
            input_dim=hidden_dim * 4,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.3
        )

        self.dropout = nn.Dropout(0.3)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, emotion_feat=None):
        x_embed = self.embedding(x)  # [B, L, D]

        if self.emotion_dim > 0 and emotion_feat is not None:
            # === Emotion Gating ===
            gate = self.emotion_gate(emotion_feat)  # [B, L, 1]
            gated_embed = x_embed * gate  # 加權輸入
            x_embed = torch.cat([gated_embed, emotion_feat], dim=-1)  # → [B, L, D+E]
        else:
            x_embed = torch.cat([x_embed, emotion_feat], dim=-1)

        x_embed = self.embedding_dropout(x_embed)

        lstm_out, _ = self.lstm(x_embed)
        lstm_out = self.lstm_norm(lstm_out)

        attn_out = self.attention(lstm_out)

        mean_pooled = attn_out.mean(dim=1)
        max_pooled = attn_out.max(dim=1).values
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)
