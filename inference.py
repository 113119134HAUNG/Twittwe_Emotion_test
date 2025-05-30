# inference.py

import torch
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_preprocessing import clean_tokens
from label_utils import load_emotion_dict, extract_emotion_index, build_sequence_emotion_features
from model import SentimentClassifier

# === 基本參數 ===
MAX_LEN = 100
SENTIMENT_CLASSES = ['Negative', 'Neutral', 'Positive']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 載入 tokenizer ===
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# === 載入情緒字典與索引 ===
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
).to(DEVICE)
model.load_state_dict(torch.load("/content/Social_IOT-NLP情緒分析/best_model.pth", map_location=DEVICE))
model.eval()

# === 單句推論函數 ===
def predict_class(text: str):
    tokens = clean_tokens(text)
    if not tokens:
        return "Unclassified", 0.0

    text_seq = tokenizer.texts_to_sequences([' '.join(tokens)])
    padded_seq = pad_sequences(text_seq, maxlen=MAX_LEN, padding='post')
    input_ids = torch.tensor(padded_seq, dtype=torch.long, device=DEVICE)

    emotion_tensor = build_sequence_emotion_features(tokens, emotion_dict, emotion2idx)
    if emotion_tensor.shape[0] < MAX_LEN:
        pad_len = MAX_LEN - emotion_tensor.shape[0]
        pad_tensor = torch.zeros((pad_len, emotion_tensor.shape[1]), device=DEVICE)
        emotion_tensor = torch.cat([emotion_tensor, pad_tensor], dim=0)
    else:
        emotion_tensor = emotion_tensor[:MAX_LEN]
    emotion_tensor = emotion_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, emotion_tensor)
        prob = torch.softmax(logits, dim=1)
        pred_idx = prob.argmax(dim=1).item()
        return SENTIMENT_CLASSES[pred_idx], float(prob[0][pred_idx])

# === 多句推論函數 ===
def predict_batch(text_list):
    results = []
    for text in text_list:
        tokens = clean_tokens(text)
        if not tokens:
            results.append((text, "Unclassified", 0.0))
            continue

        token_str = ' '.join(tokens)
        input_ids = tokenizer.texts_to_sequences([token_str])
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, padding='post')
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)

        emotion_tensor = build_sequence_emotion_features(tokens, emotion_dict, emotion2idx)
        if emotion_tensor.shape[0] < MAX_LEN:
            pad_tensor = torch.zeros((MAX_LEN - emotion_tensor.shape[0], emotion_dim), device=DEVICE)
            emotion_tensor = torch.cat([emotion_tensor, pad_tensor], dim=0)
        else:
            emotion_tensor = emotion_tensor[:MAX_LEN]
        emotion_tensor = emotion_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor, emotion_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            label = SENTIMENT_CLASSES[pred_idx]
            confidence = probs[0][pred_idx].item()
            results.append((text, label, confidence))

    return results

# === CLI 測試 ===
if __name__ == "__main__":
    while True:
        text = input("Enter sentence (or 'exit'): ").strip()
        if text.lower() == 'exit':
            break
        label, confidence = predict_class(text)
        print(f"→ 預測類別：{label} (信心：{confidence:.2%})")
