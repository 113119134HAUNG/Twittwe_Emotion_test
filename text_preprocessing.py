# text_preprocessing.py

import re
import emoji
import random
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from contractions_dict import contractions
from stopword_set import stop_words

nltk.download('stopwords', quiet=True)
tokenizer = TreebankWordTokenizer()

# === 共用文字標準化處理 ===
def text_standardize(text: str) -> str:
    text = text.lower()
    text = expand_contractions(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# === 擴展縮寫詞 ===
def expand_contractions(text: str) -> str:
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions) + r')\b', flags=re.IGNORECASE)
    return pattern.sub(lambda m: contractions[m.group().lower()], text)

# === 模擬社群雜訊：隨機字母錯位 ===
def random_typo(word: str) -> str:
    if len(word) > 3 and random.random() < 0.3:
        chars = list(word)
        i = random.randint(1, len(word) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return ''.join(chars)
    return word

# === 隨機加入 emoji 到單字 ===
def add_emoji(word: str) -> str:
    if random.random() < 0.3:
        return word + " 😊"
    return word

# === 隨機加入 emoji 到句尾 ===
def add_emoji_noise(text: str) -> str:
    emojis = ['😊', '😐', '😢', '😂', '😡', '👍', '🙏', '💬', '🔥']
    if random.random() < 0.2:
        return text + " " + random.choice(emojis)
    return text

# === 隨機加入語氣詞 ===
def add_tone_words(text: str) -> str:
    tones = ['really', 'actually', 'seriously', 'totally']
    if random.random() < 0.2:
        return text + " " + random.choice(tones)
    return text

# === 綜合擾動文字增強 ===
def text_augment(text: str) -> str:
    text = add_emoji_noise(text)
    text = add_tone_words(text)
    return text

# === 訓練用預處理：加噪版本 ===
def advanced_clean(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text_standardize(text)
    text = text_augment(text)
    tokens = tokenizer.tokenize(text)
    return [add_emoji(random_typo(t)) for t in tokens if t not in stop_words and len(t) > 1]

# === 推論用預處理：乾淨版本 ===
def clean_tokens(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text_standardize(text)
    tokens = tokenizer.tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 1]
