# run_analysis.py

import os
import json
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
from label_utils import classify_tokens, load_emotion_dict
from text_preprocessing import clean_tokens

# === 初始化 ===
console = Console()
emotion_dict = load_emotion_dict("NRC_Emotion_Label.csv")
label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}

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

    for sent in sentences:
        tokens = clean_tokens(sent)

        if not tokens:
            label = 0  # 預設 Neutral
        else:
            label = classify_tokens(tokens, emotion_dict)
            if label is None:
                label = 0  # 無法分類則視為中性

        label_str = label_map.get(label, "Unknown")

        table = Table(title="單句情緒預測結果", show_lines=True)
        table.add_column("Text", style="cyan")
        table.add_column("Predicted Emotion", style="green")
        table.add_row(sent, label_str)
        console.print(table)

        rows.append({
            'date': date,
            'section': section,
            'sentence': sent,
            'emotion': label_str,
            'confidence': 1.0
        })

    return pd.DataFrame(rows)


def analyze_diary(diary: dict):
    date = format_date(diary.get('date', ''))
    content = diary.get('sections', {})
    all_df = pd.DataFrame()

    for section in ['emotion_source', 'process_description', 'emotion_reflection', 'emotion_tracking']:
        text = content.get(section, {}).get('content', '')
        if text:
            section_df = analyze_emotions(text, section, date)
            all_df = pd.concat([all_df, section_df], ignore_index=True)

    return all_df


def analyze_all(json_dir="./diary_json", output_csv="Dict_all_diary_analyses.csv"):
    if not os.path.exists(json_dir):
        console.print(f"[red]找不到目錄：{json_dir}[/red]")
        return

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    if not json_files:
        console.print("[red]未找到任何 JSON 檔案。[/red]")
        return

    all_results = pd.DataFrame()

    for file in json_files:
        console.print(f"[bold yellow]\n分析中: {file}[/bold yellow]")
        try:
            with open(os.path.join(json_dir, file), encoding='utf-8') as f:
                data = json.load(f)
            result = analyze_diary(data)
            result['file'] = file
            all_results = pd.concat([all_results, result], ignore_index=True)
        except Exception as e:
            console.print(f"[red]錯誤處理 {file}: {e}[/red]")

    if not all_results.empty:
        all_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
        console.print(f"[green bold]\n儲存成功: {output_csv}")
        console.print(f"總日記數: {len(json_files)}")
        console.print(f"總句子數: {len(all_results)}")

        console.print("\n[blue bold]整體情緒分佈:[/blue bold]")
        for emo, count in all_results['emotion'].value_counts().items():
            console.print(f"- {emo}: {count} ({count/len(all_results)*100:.1f}%)")
    else:
        console.print("[red bold]無分析結果。[/red bold]")


if __name__ == '__main__':
    analyze_all()
