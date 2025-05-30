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

def expand_contractions(text):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions) + r')\b', flags=re.IGNORECASE)
    return pattern.sub(lambda m: contractions[m.group().lower()], text)

def random_typo(word):
    if len(word) > 3 and random.random() < 0.3:
        chars = list(word)
        i = random.randint(1, len(word)-2)
        chars[i], chars[i+1] = chars[i+1], chars[i]
        return ''.join(chars)
    return word

def add_emoji(word):
    if random.random() < 0.3:
        return word + " 😊"
    return word

def add_emoji_noise(text):
    emojis = ['😊', '😐', '😢']
    if random.random() < 0.2:
        return text + " " + random.choice(emojis)
    return text

def add_tone_words(text):
    if random.random() < 0.2:
        return text + " really"  # 模擬強調詞
    return text

def text_augment(text):
    text = add_emoji_noise(text)
    text = add_tone_words(text)
    return text

# === 資料增強用：隨機加入 typo / emoji 模擬社群雜訊 =
def advanced_clean(text: str) -> list[str]:
    if not isinstance(text, str): return []
    text = text.lower()
    text = expand_contractions(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = tokenizer.tokenize(text)
    # 資料增強
    return [add_emoji(random_typo(t))for t in tokens if t not in stop_words and len(t) > 1]

# === 推論用：乾淨無擾動版本 ===
def clean_tokens(text: str) -> list[str]:
    if not isinstance(text, str): return []
    text = text.lower()
    text = expand_contractions(text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = tokenizer.tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 1]