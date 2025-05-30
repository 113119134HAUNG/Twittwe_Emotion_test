# diary_analysis.py

import os
import torch
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences

from text_preprocessing import advanced_clean,clean_tokens
from model import SentimentClassifier
# 設定
sentiment_classes = ['Negative', 'Neutral', 'Positive']
max_length = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 tokenizer 與模型
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.pickle")
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

model = SentimentClassifier().to(device)
model.load_state_dict(torch.load("/content/Social_IOT-NLP情緒分析/best_model.pth", map_location=device))
model.eval()

def analyze_emotions(text, tokenizer, model, max_length, date=None):
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    cleaned = [' '.join(clean_tokens(s)) for s in sentences]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    input_tensor = torch.tensor(padded).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_indices = probs.argmax(dim=1).cpu().numpy()
        confidences = probs.max(dim=1).values.cpu().numpy()

    results = []
    for i, sentence in enumerate(sentences):
        results.append({
            'date': date,
            'sentence': sentence,
            'emotion': sentiment_classes[pred_indices[i]],
            'confidence': float(confidences[i])
        })
    return pd.DataFrame(results)

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

def save_results(df, output_path='diary_emotion_analysis.xlsx'):
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        print(f"✅ 結果已儲存至 {output_path}")
    except Exception as e:
        print("⚠️ Excel 儲存失敗，改存為 CSV。", str(e))
        df.to_csv('diary_emotion_analysis.csv', index=False, encoding='utf-8-sig')
        print("備援：已儲存為 diary_emotion_analysis.csv")

def main():
    with open("/content/Social IOT-NLP情緒分析/diary.txt", encoding='utf-8') as f:
        text = f.read()

    results_df = analyze_emotions(text, tokenizer, model, max_length)
    visualize_results(results_df)
    save_results(results_df)

if __name__ == '__main__':
    main()