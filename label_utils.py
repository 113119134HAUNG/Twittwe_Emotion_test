# label_utils.py

import re
import torch
import pandas as pd
import numpy as np
from collections import Counter
from typing import List
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import wordnet

# 外部資源
from filler_words import filler_words, single_word_fillers, phrase_fillers, phrase_emotion_dict
from contractions_dict import contractions
from negation_words import negation_words
from intensifier_words import intensifier_words
from text_preprocessing import advanced_clean, clean_tokens

# === 載入已處理好的情緒詞典 ===
def load_emotion_dict(path: str = "NRC_Emotion_Label.csv") -> dict:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['word', 'emotion', 'association'])
    emotion_dict = {}
    for _, row in df.iterrows():
        word = str(row['word']).strip().lower()
        label = int(row['association'])
        tags = [t.strip() for t in str(row['emotion']).split(',') if t.strip()]
        emotion_dict[word] = {"label": label, "tags": tags}
    return emotion_dict

# === 建立 emotion → index 對照表 ===
def extract_emotion_index(emotion_dict: dict) -> dict:
    all_tags = set(tag for entry in emotion_dict.values() for tag in entry["tags"])
    return {tag: idx for idx, tag in enumerate(sorted(all_tags))}

# === 正規化縮小語 ===
def normalize_token(word: str) -> str:
    return contractions.get(word.lower().strip(), word.lower().strip())

# === 判斷 token 是否為有效詞彙（WordNet） ===
def is_valid_token(token: str) -> bool:
    return token.isalpha() and wordnet.synsets(token)

# === 清洗並過濾句子（回傳 None 表示該句被丟棄）===
def clean_and_filter_tokens(text: str) -> list[str] | None:
    tokens = clean_tokens(text)
    valid_tokens = [t for t in tokens if is_valid_token(t)]
    return valid_tokens if len(valid_tokens) >= 5 else None

# === 主分類函數（詞組極性） ===
def classify_tokens(tokens: List[str], emotion_dict: dict) -> int | None:
    labels = []
    negation_count = 0
    intensify_weight = 1.0
    i = 0

    while i < len(tokens):
        word = normalize_token(tokens[i])

        if not is_valid_token(word):
            i += 1
            continue

        if word in single_word_fillers:
            i += 1
            continue

        # 處理三詞短語
        if i + 2 < len(tokens):
            tri_phrase = ' '.join([tokens[i], tokens[i+1], tokens[i+2]])
            if tri_phrase in phrase_emotion_dict:
                phrase_info = phrase_emotion_dict[tri_phrase]
                label = phrase_info['label']
                if negation_count % 2 == 1:
                    label = -label
                labels.append(label)
                negation_count = 0
                intensify_weight = 1.0
                i += 3
                continue

        # 處理二詞短語
        if i + 1 < len(tokens):
            bi_phrase = ' '.join([tokens[i], tokens[i+1]])
            if bi_phrase in phrase_emotion_dict:
                phrase_info = phrase_emotion_dict[bi_phrase]
                label = phrase_info['label']
                if negation_count % 2 == 1:
                    label = -label
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

        if word in emotion_dict:
            label = emotion_dict[word]['label']
            if negation_count % 2 == 1:
                label = -label
            label = int(round(label * intensify_weight))
            labels.append(label)
            negation_count = 0
            intensify_weight = 1.0

        i += 1

    return Counter(labels).most_common(1)[0][0] if labels else None

# === 單字轉 multi-hot 向量 ===
def build_emotion_feature(word: str, emotion_dict: dict, emotion2idx: dict) -> np.ndarray:
    tags = []

    # 單字情緒標籤
    if word.lower() in emotion_dict:
        tags += emotion_dict[word.lower()].get("tags", [])

    # 如果是短語（bi/tri-phrase），也檢查 phrase_emotion_dict
    if word.lower() in phrase_emotion_dict:
        tags += phrase_emotion_dict[word.lower()].get("tags", [])
        tags += phrase_emotion_dict[word.lower()].get("emotion_tags", [])

    vec = np.zeros(len(emotion2idx), dtype=int)
    for tag in tags:
        if tag in emotion2idx:
            vec[emotion2idx[tag]] = 1
    return vec

# === 整個 token 序列的向量化表示 ===
def build_sequence_emotion_features(tokens: List[str], emotion_dict: dict, emotion2idx: dict) -> torch.Tensor:
    vectors = [build_emotion_feature(t, emotion_dict, emotion2idx) for t in tokens]
    return torch.tensor(vectors, dtype=torch.float)

# === 輔助函數：輸出極性與 multi-hot tensor ===
def classify_with_feature(tokens: List[str], emotion_dict: dict, emotion2idx: dict):
    polarity = classify_tokens(tokens, emotion_dict)
    emotion_tensor = build_sequence_emotion_features(tokens, emotion_dict, emotion2idx)
    return polarity, emotion_tensor