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

# === 基本參數 ===
sentiment_classes = ['Negative', 'Neutral', 'Positive']
max_length = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 載入 tokenizer ===
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.pickle")
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

# === 載入情緒字典 ===
emotion_dict = load_emotion_dict("/content/Social_IOT-NLP情緒分析/NRC_Emotion_Label.csv")
emotion2idx = extract_emotion_index(emotion_dict)
emotion_dim = len(emotion2idx)

# === 載入模型 ===
model = SentimentClassifier(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=128,
    num_heads=4,
    num_classes=3,
    emotion_dim=emotion_dim
).to(device)
model.load_state_dict(torch.load("/content/Social_IOT-NLP情緒分析/best_model.pth", map_location=device))
model.eval()

# === 單篇日記分析 ===
def analyze_emotions(text, tokenizer, model, max_length, date=None):
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    results = []

    for sentence in sentences:
        tokens = clean_tokens(sentence)
        if not tokens:
            emotion = "Neutral"
            confidence = 1.0
        else:
            token_str = ' '.join(tokens)
            token_ids = tokenizer.texts_to_sequences([token_str])
            token_ids = pad_sequences(token_ids, maxlen=max_length, padding='post')
            input_tensor = torch.tensor(token_ids).to(device)

            emotion_tensor = build_sequence_emotion_features(tokens, emotion_dict, emotion2idx)
            if emotion_tensor.shape[0] < max_length:
                pad_len = max_length - emotion_tensor.shape[0]
                pad_tensor = torch.zeros((pad_len, emotion_dim))
                emotion_tensor = torch.cat([emotion_tensor, pad_tensor], dim=0)
            else:
                emotion_tensor = emotion_tensor[:max_length]
            emotion_tensor = emotion_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor, emotion_tensor)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).item()
                confidence = probs[0][pred].item()
                emotion = sentiment_classes[pred]

        results.append({
            'date': date,
            'sentence': sentence,
            'emotion': emotion,
            'confidence': confidence
        })

    return pd.DataFrame(results)

# === 可視化結果 ===
def visualize_results(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 圓餅圖
    emotion_counts = df['emotion'].value_counts()
    ax1.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%')
    ax1.set_title('情緒分布')

    # 信心值箱型圖
    sns.boxplot(x='emotion', y='confidence', data=df, ax=ax2)
    ax2.set_title('信心值分佈')

    plt.tight_layout()
    plt.show()

# === 儲存分析結果 ===
def save_results(df, output_path='diary_emotion_analysis.xlsx'):
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        print(f"結果已儲存至 {output_path}")
    except Exception as e:
        print("⚠️ Excel 儲存失敗，改存為 CSV。", str(e))
        df.to_csv('diary_emotion_analysis.csv', index=False, encoding='utf-8-sig')
        print("備援：已儲存為 diary_emotion_analysis.csv")

# === 主執行 ===
def main():
    with open("/content/Social IOT-NLP情緒分析/diary.txt", encoding='utf-8') as f:
        text = f.read()

    results_df = analyze_emotions(text, tokenizer, model, max_length)
    visualize_results(results_df)
    save_results(results_df)

if __name__ == '__main__':
    main()
