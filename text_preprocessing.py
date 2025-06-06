# text_preprocessing.py

import re
import emoji
import random
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from contractions_dict import contractions
from stopword_set import stop_words

nltk.download('stopwords', quiet=True)

# === 標準化處理 ===
def text_standardize(text: str) -> str:
    text = text.lower()
    text = expand_contractions(text)
    text = emoji.demojize(text, delimiters=(" ", " "))  # 😄 → :smile:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s:]", " ", text)  # 保留 :smile: 表情
    return re.sub(r"\s+", " ", text).strip()

# === 擴展縮寫詞 ===
def expand_contractions(text: str) -> str:
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions) + r')\b', flags=re.IGNORECASE)
    return pattern.sub(lambda m: contractions[m.group().lower()], text)

# === 拼寫錯誤擾動（微調）===
def random_typo(word: str) -> str:
    if len(word) > 3 and random.random() < 0.2:
        chars = list(word)
        i = random.randint(1, len(word) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return ''.join(chars)
    return word

# === Emoji 隨機插入（輕微）===
def add_emoji_noise(text: str) -> str:
    emojis = [':smile:', ':neutral_face:', ':cry:', ':thumbs_up:', ':fire:']
    if random.random() < 0.15:
        return text + " " + random.choice(emojis)
    return text

# === 語氣詞插入 ===
def add_tone_words(text: str) -> str:
    tones = ['really', 'actually', 'seriously', 'totally']
    if random.random() < 0.2:
        return text + " " + random.choice(tones)
    return text

# === 綜合增強 ===
def text_augment(text: str) -> str:
    text = add_emoji_noise(text)
    text = add_tone_words(text)
    return text

# === 訓練用清洗（有噪聲）===
def advanced_clean(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text_standardize(text)
    text = text_augment(text)
    tokens = wordpunct_tokenize(text)
    return [random_typo(t) for t in tokens if t not in stop_words and len(t) > 1]

# === 推論用清洗（乾淨）===
def clean_tokens(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text_standardize(text)
    tokens = wordpunct_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 1]
