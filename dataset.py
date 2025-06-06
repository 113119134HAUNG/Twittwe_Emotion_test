# dataset.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from label_utils import build_sequence_emotion_features

class SentimentDataset(Dataset):
    def __init__(self, token_texts, y_class, tokenizer, emotion_dict, emotion2idx,
                 sources=None, max_len=100):
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
        text_str = ' '.join(tokens)

        # === Token IDs ===
        token_ids = self.tokenizer.texts_to_sequences([text_str])[0] or [0]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        # === Padding token IDs (with cutoff) ===
        input_ids = token_tensor[:self.max_len]
        input_ids = F.pad(input_ids, (0, self.max_len - len(input_ids)), value=0)

        # === Emotion Features ===
        with torch.no_grad():
            emotion_feat = build_sequence_emotion_features(tokens, self.emotion_dict, self.emotion2idx)

        padded_emotion = emotion_feat[:self.max_len]
        pad_len = self.max_len - padded_emotion.shape[0]
        padded_emotion = F.pad(padded_emotion, (0, 0, 0, pad_len))  # → [max_len, D]

        # === Result Output ===
        result = {
            'input_ids': input_ids,
            'label': torch.tensor(int(self.y_class[idx]), dtype=torch.long),
            'emotion_feat': padded_emotion,
            'length': min(len(token_tensor), self.max_len)  # 可選長度（可用於 attention mask）
        }

        if self.sources is not None:
            result['source'] = self.sources[idx]

        return result
