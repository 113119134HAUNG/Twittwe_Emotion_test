# label_utils.py

import re
import ast
import torch
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Optional, Union
from functools import lru_cache
from nltk.tokenize import wordpunct_tokenize

# 外部資源
from filler_words import filler_words, single_word_fillers, phrase_fillers, phrase_emotion_dict
from contractions_dict import contractions
from negation_words import negation_words
from intensifier_words import intensifier_words
from text_preprocessing import advanced_clean, clean_tokens

# === 載入情緒字典（含中性補充與污染檢查）===
def load_emotion_dict(path: str = "NRC_Emotion_Label.csv", neutral_path: Optional[str] = None) -> dict:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['word', 'emotion', 'association'])
    emotion_dict = {}

    for _, row in df.iterrows():
        word = str(row['word']).strip().lower()
        try:
            label = int(row['association'])
            tags = [t.strip() for t in str(row['emotion']).split(',') if t.strip()]
            emotion_dict[word] = {"label": label, "tags": tags}
        except Exception as e:
            print(f"跳過詞彙 '{word}'（格式錯誤）: {e}")

    if neutral_path:
        try:
            df_neu = pd.read_csv(neutral_path)
            df_neu.columns = df_neu.columns.str.strip()
            df_neu = df_neu.dropna(subset=['word'])
            for word in df_neu['word'].astype(str).str.strip().str.lower().unique():
                if word not in emotion_dict or not isinstance(emotion_dict[word], dict):
                    emotion_dict[word] = {"label": 0, "tags": ["neutral"]}
        except Exception as e:
            print(f"中性詞典載入錯誤: {e}")

    polluted = [k for k, v in emotion_dict.items() if not isinstance(v, dict)]
    if polluted:
        print(f"有 {len(polluted)} 筆詞彙被污染為非 dict（例如 int），前 5 筆：{polluted[:5]}")

    return emotion_dict

# === 載入中性詞集合 ===
def load_neutral_dict(path: str = "NRC_Emotion_Label2.csv") -> set:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['word', 'emotion', 'association'])
    neutral_words = df[
        (df['association'].astype(int) == 0) |
        (df['emotion'].str.contains('neutral', case=False, na=False))
    ]['word'].str.strip().str.lower()
    return set(neutral_words)

# === 建立 emotion → index 對照表 ===
def extract_emotion_index(emotion_dict: dict) -> dict:
    all_tags = set(tag for entry in emotion_dict.values() if isinstance(entry, dict) for tag in entry.get("tags", []))
    return {tag: idx for idx, tag in enumerate(sorted(all_tags))}

# === 正規化縮寫 ===
@lru_cache(maxsize=10000)
def normalize_token(word: str) -> str:
    return contractions.get(word.lower().strip(), word.lower().strip())

# === 有效詞判斷 ===
@lru_cache(maxsize=50000)
def is_valid_token(token: str) -> bool:
    return token.isalpha()

# === 清理並保留有效詞 ===
def clean_and_filter_tokens(text: str) -> Optional[List[str]]:
    tokens = clean_tokens(text)
    valid_tokens = [t for t in tokens if is_valid_token(t)]
    return valid_tokens if len(valid_tokens) >= 5 else None

# === 主分類函數 ===
def classify_tokens(tokens: List[str], emotion_dict: dict, return_counts=False) -> Union[int, dict, None]:
    labels = []
    negation_count = 0
    intensify_weight = 1.0
    i = 0

    while i < len(tokens):
        word = normalize_token(tokens[i])

        if not is_valid_token(word) or word in single_word_fillers:
            i += 1
            continue

        if i + 2 < len(tokens):
            tri = ' '.join([tokens[i], tokens[i+1], tokens[i+2]])
            if tri in phrase_emotion_dict:
                label = phrase_emotion_dict[tri].get('label')
                if isinstance(label, int):
                    if negation_count % 2 == 1:
                        label = -label
                    label = int(np.clip(round(label * intensify_weight), -1, 1))
                    labels.append(label)
                negation_count = 0
                intensify_weight = 1.0
                i += 3
                continue

        if i + 1 < len(tokens):
            bi = ' '.join([tokens[i], tokens[i+1]])
            if bi in phrase_emotion_dict:
                label = phrase_emotion_dict[bi].get('label')
                if isinstance(label, int):
                    if negation_count % 2 == 1:
                        label = -label
                    label = int(np.clip(round(label * intensify_weight), -1, 1))
                    labels.append(label)
                negation_count = 0
                intensify_weight = 1.0
                i += 2
                continue

        if word in negation_words:
            negation_count += 1
            i += 1
            continue

        if word in intensifier_words:
            intensify_weight = intensifier_words[word]
            i += 1
            continue

        if word in emotion_dict and isinstance(emotion_dict[word], dict):
            label = emotion_dict[word].get("label", None)  # 不預設為 0
            if isinstance(label, int):
                if negation_count % 2 == 1:
                    label = -label
                label = int(np.clip(round(label * intensify_weight), -1, 1))
                labels.append(label)
            negation_count = 0
            intensify_weight = 1.0

        i += 1

    if not labels:
        # 如果都沒分類出來，但含有標記為中性的詞
        if any(
            word in emotion_dict and isinstance(emotion_dict[word], dict) and emotion_dict[word].get("label") == 0
            for word in tokens
        ):
            return 0 if not return_counts else {0: 1}
        return None if not return_counts else {}

    counts = Counter(labels)
    return counts.most_common(1)[0][0] if not return_counts else dict(counts)

# === 使用雙字典（中性詞單獨處理）分類 ===
def classify_with_two_dicts(tokens: List[str], emotion_dict: dict, neutral_dict: set) -> Optional[int]:
    pos, neg, neu = 0, 0, 0
    for token in tokens:
        word = normalize_token(token.lower())
        if not is_valid_token(word):
            continue
        if word in neutral_dict:
            neu += 1
        elif word in emotion_dict and isinstance(emotion_dict[word], dict):
            label = emotion_dict[word].get('label', 0)
            if label == 1:
                pos += 1
            elif label == -1:
                neg += 1

    if pos > max(neg, neu):
        return 1
    elif neg > max(pos, neu):
        return -1
    elif neu > 0 and pos == 0 and neg == 0:
        return 0
    else:
        return None

# === 向量化：單字情緒向量 ===
def build_emotion_feature(word: str, emotion_dict: dict, emotion2idx: dict) -> np.ndarray:
    word = word.lower()
    tags = []

    if word in emotion_dict and isinstance(emotion_dict[word], dict):
        tags += emotion_dict[word].get("tags", [])

    if word in phrase_emotion_dict:
        tags += phrase_emotion_dict[word].get("tags", [])
        tags += phrase_emotion_dict[word].get("emotion_tags", [])

    vec = np.zeros(len(emotion2idx), dtype=int)
    for tag in tags:
        if tag in emotion2idx:
            vec[emotion2idx[tag]] = 1

    if vec.sum() == 0 and 'neutral' in emotion2idx:
        vec[emotion2idx['neutral']] = 1

    return vec

# === 向量化：整句情緒向量 ===
def build_sequence_emotion_features(tokens: List[str], emotion_dict: dict, emotion2idx: dict) -> torch.Tensor:
    vectors = [build_emotion_feature(t, emotion_dict, emotion2idx) for t in tokens]
    return torch.tensor(vectors, dtype=torch.float)

# === 同時取得分類與向量 ===
def classify_with_feature(tokens: List[str], emotion_dict: dict, emotion2idx: dict):
    polarity = classify_tokens(tokens, emotion_dict)
    emotion_tensor = build_sequence_emotion_features(tokens, emotion_dict, emotion2idx)
    return polarity, emotion_tensor
