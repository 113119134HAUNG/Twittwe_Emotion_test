# run_analysis.py

import os
import json
import pandas as pd
from datetime import datetime
from rich import print as rprint
from rich.table import Table
from nltk.tokenize import wordpunct_tokenize

# 自定義模組
from label_utils import classify_tokens, load_emotion_dict
from text_preprocessing import advanced_clean,clean_tokens

# === 載入字典 ===
emotion_dict = load_emotion_dict("NRC_Emotion_Label.csv")


def format_date(date_str):
    try:
        try:
            return datetime.strptime(date_str, '%B %d, %Y').strftime('%Y/%m/%d')
        except:
            digits = ''.join(filter(str.isdigit, date_str))
            if len(digits) == 8:
                return f"{digits[:4]}/{digits[4:6]}/{digits[6:]}"
        return datetime.now().strftime('%Y/%m/%d')
    except:
        return datetime.now().strftime('%Y/%m/%d')


def analyze_emotions(text, section, date):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    rows = []
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}

    for sent in sentences:
        clean = clean_tokens(sent)
        tokens = wordpunct_tokenize(clean)
        label = classify_tokens(tokens, emotion_dict)
        label_str = label_map.get(label, "Unknown")

        table = Table(title="單句情緒預測結果", show_lines=True)
        table.add_column("Text", style="cyan")
        table.add_column("Predicted Emotion", style="green")
        rprint(table)
        table.add_row(sent, label_str)

        rows.append({
            'date': date,
            'section': section,
            'sentence': sent,
            'emotion': label_str,
            'confidence': 1.0
        })
    return pd.DataFrame(rows)


def analyze_diary(diary):
    date = format_date(diary.get('date', ''))
    content = diary.get('sections', {})
    all_df = pd.DataFrame()
    for section in ['emotion_source', 'process_description', 'emotion_reflection', 'emotion_tracking']:
        text = content.get(section, {}).get('content', '')
        if text:
            section_df = analyze_emotions(text, section, date)
            all_df = pd.concat([all_df, section_df], ignore_index=True)
    return all_df


def analyze_all():
    path = "./diary_json"
    json_files = [f for f in os.listdir(path) if f.endswith(".json")]
    all_results = pd.DataFrame()

    for file in json_files:
        rprint(f"[bold yellow]\n分析中: {file}[/bold yellow]")
        with open(os.path.join(path, file), encoding='utf-8') as f:
            data = json.load(f)
        result = analyze_diary(data)
        result['file'] = file
        all_results = pd.concat([all_results, result], ignore_index=True)

    if not all_results.empty:
        all_results.to_csv("Dict_all_diary_analyses.csv", index=False, encoding='utf-8-sig')
        rprint("[green bold]\n儲存成功: Dict_all_diary_analyses.csv")
        rprint(f"總日記數: {len(json_files)}")
        rprint(f"總句子數: {len(all_results)}")

        rprint("\n[blue bold]整體情緒分佈:[/blue bold]")
        for emo, count in all_results['emotion'].value_counts().items():
            rprint(f"- {emo}: {count} ({count/len(all_results)*100:.1f}%)")
    else:
        rprint("[red bold]無分析結果。[/red bold]")


if __name__ == '__main__':
    analyze_all()