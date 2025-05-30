# dataset.py

import torch
from torch.utils.data import Dataset
from label_utils import build_sequence_emotion_features

class SentimentDataset(Dataset):
    def __init__(self, token_texts, y_class, tokenizer, emotion_dict, emotion2idx, sources=None, max_len=100):
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
        tokens = self.token_texts[idx] or ["[PAD]"]

        with torch.no_grad():
            emotion_feat = build_sequence_emotion_features(tokens, self.emotion_dict, self.emotion2idx)

        text_str = ' '.join(tokens)
        token_ids = self.tokenizer.texts_to_sequences([text_str])[0] or [0]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        # Padding input_ids
        if len(token_tensor) < self.max_len:
            input_ids = torch.cat([token_tensor, torch.zeros(self.max_len - len(token_tensor), dtype=torch.long)])
        else:
            input_ids = token_tensor[:self.max_len]

        # Padding emotion features
        padded_emotion = emotion_feat[:self.max_len]
        if len(padded_emotion) < self.max_len:
            pad_size = self.max_len - len(padded_emotion)
            pad_tensor = torch.zeros((pad_size, len(self.emotion2idx)), dtype=torch.float32)
            padded_emotion = torch.cat([padded_emotion, pad_tensor], dim=0)

        result = {
            'input_ids': input_ids,
            'label': torch.tensor(int(self.y_class[idx]), dtype=torch.long),
            'emotion_feat': padded_emotion
        }

        if self.sources is not None:
            result['source'] = self.sources[idx]

        return result
