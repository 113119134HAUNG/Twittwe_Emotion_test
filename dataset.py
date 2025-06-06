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

        # === Padding token IDs ===
        input_ids = F.pad(token_tensor, (0, self.max_len - len(token_tensor)), value=0)
        input_ids = input_ids[:self.max_len]  # 防止超過 max_len

        # === Emotion Features ===
        with torch.no_grad():
            emotion_feat = build_sequence_emotion_features(tokens, self.emotion_dict, self.emotion2idx)

        pad_len = self.max_len - emotion_feat.shape[0]
        padded_emotion = F.pad(emotion_feat, (0, 0, 0, pad_len))  # [L, D] → [max_len, D]
        padded_emotion = padded_emotion[:self.max_len]

        # === Output Dict ===
        result = {
            'input_ids': input_ids,
            'label': torch.tensor(int(self.y_class[idx]), dtype=torch.long),
            'emotion_feat': padded_emotion,
            'length': min(len(token_tensor), self.max_len)  # 可選輸出：原始長度
        }

        if self.sources is not None:
            result['source'] = self.sources[idx]

        return result
