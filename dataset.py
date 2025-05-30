# dataset.py

import torch
from torch.utils.data import Dataset
from label_utils import build_sequence_emotion_features

class SentimentDataset(Dataset):
    def __init__(self, token_texts, y_class, tokenizer, emotion_dict, emotion2idx, sources, max_len=100):
        self.token_texts = token_texts
        self.y_class = y_class
        self.sources = sources
        self.tokenizer = tokenizer
        self.emotion_dict = emotion_dict
        self.emotion2idx = emotion2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.token_texts)

    def __getitem__(self, idx):
        tokens = self.token_texts[idx]
        emotion_feat = build_sequence_emotion_features(tokens, self.emotion_dict, self.emotion2idx)

        text_str = ' '.join(tokens)
        token_ids = self.tokenizer.texts_to_sequences([text_str])[0]
        if not token_ids:
            token_ids = [0]

        padded_ids = token_ids[:self.max_len] + [0] * max(0, self.max_len - len(token_ids))

        padded_emotion = emotion_feat[:self.max_len]
        if len(padded_emotion) < self.max_len:
            pad_size = self.max_len - len(padded_emotion)
            pad_tensor = torch.zeros((pad_size, len(self.emotion2idx)), dtype=torch.float)
            padded_emotion = torch.cat([padded_emotion, pad_tensor], dim=0)

        return {
            'input_ids': torch.tensor(padded_ids, dtype=torch.long),
            'label': torch.tensor(int(self.y_class[idx]), dtype=torch.long),
            'emotion_feat': padded_emotion,
            'source': self.sources[idx]
        }