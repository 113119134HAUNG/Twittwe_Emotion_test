# diary_analysis.py

import os
import torch
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences

from text_preprocessing import clean_tokens
from model import SentimentClassifier
from label_utils import load_emotion_dict, extract_emotion_index, build_sequence_emotion_features

# === 設定參數 ===
SENTIMENT_CLASSES = ['Negative', 'Neutral', 'Positive']
MAX_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 設定路徑（可透過環境變數調整）===
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.pickle")
DICT_PATH = os.getenv("EMOTION_DICT", "NRC_Emotion_Label.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")
DIARY_PATH = os.getenv("DIARY_PATH", "diary.txt")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "diary_emotion_analysis.xlsx")

# === 載入 tokenizer ===
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"找不到 tokenizer 檔案：{TOKENIZER_PATH}")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# === 載入字典與模型 ===
emotion_dict = load_emotion_dict(DICT_PATH)
emotion2idx = extract_emotion_index(emotion_dict)
emotion_dim = len(emotion2idx)

model = SentimentClassifier(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=128,
    num_heads=4,
    num_classes=3,
    emotion_dim=emotion_dim
).to(DEVICE)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def pad_emotion_tensor(tensor, max_len, dim):
    """補齊情緒張量長度"""
    if tensor.shape[0] < max_len:
        pad_len = max_len - tensor.shape[0]
        pad_tensor = torch.zeros((pad_len, dim))
        return torch.cat([tensor, pad_tensor], dim=0)
    return tensor[:max_len]


def analyze_emotions(text, date=None):
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    results = []

    for sentence in sentences:
        tokens = clean_tokens(sentence)
        if not tokens:
            emotion, confidence = "Neutral", 1.0
        else:
            token_ids = tokenizer.texts_to_sequences([' '.join(tokens)])
            token_ids = pad_sequences(token_ids, maxlen=MAX_LEN, padding='post')
            input_tensor = torch.tensor(token_ids, dtype=torch.long).to(DEVICE)

            emotion_tensor = build_sequence_emotion_features(tokens, emotion_dict, emotion2idx)
            emotion_tensor = pad_emotion_tensor(emotion_tensor, MAX_LEN, emotion_dim).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_tensor, emotion_tensor)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).item()
                emotion = SENTIMENT_CLASSES[pred]
                confidence = probs[0][pred].item()

        results.append({
            'date': date,
            'sentence': sentence,
            'emotion': emotion,
            'confidence': confidence
        })

    return pd.DataFrame(results)


def visualize_results(df):
    plt.figure(figsize=(14, 5))

    # 子圖 1：情緒分布圓餅圖
    plt.subplot(1, 2, 1)
    counts = df['emotion'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("情緒分布")

    # 子圖 2：信心值箱型圖
    plt.subplot(1, 2, 2)
    sns.boxplot(x='emotion', y='confidence', data=df)
    plt.title("信心值分布")

    plt.tight_layout()
    plt.show()

def save_results(df, path=OUTPUT_PATH):
    try:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        print(f"分析結果已儲存至：{path}")
    except Exception as e:
        fallback_path = "diary_emotion_analysis.csv"
        df.to_csv(fallback_path, index=False, encoding='utf-8-sig')
        print(f" Excel 儲存失敗，已改存為 CSV：{fallback_path}\n錯誤訊息：{e}")

def main():
    if not os.path.exists(DIARY_PATH):
        raise FileNotFoundError(f"找不到日記檔案：{DIARY_PATH}")

    with open(DIARY_PATH, encoding='utf-8') as f:
        text = f.read()

    results_df = analyze_emotions(text)
    visualize_results(results_df)
    save_results(results_df)

if __name__ == '__main__':
    main()
