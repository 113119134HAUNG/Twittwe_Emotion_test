# train.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import numpy as np

# 全域紀錄（可視覺化訓練曲線）
train_loss_list = []
train_acc_list = []
train_precision_list = []
train_recall_list = []
train_f1_list = []

def train_one_epoch(model, train_loader, class_weights_tensor, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_trues = [], []

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)
    for batch in loop:
        x = batch['input_ids'].to(device)
        y_class = batch['label'].to(device)
        emotion_tensor = batch['emotion_feat'].to(device)
        sources = batch['source']

        optimizer.zero_grad()
        logits = model(x, emotion_tensor)
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)

        # One-hot labels: [B, C]
        y_onehot = torch.nn.functional.one_hot(y_class, num_classes=logits.size(1)).float()

        # 取得 class weights
        class_weight_per_sample = class_weights_tensor[y_class].clone()
        neutral_mask = (y_class == 1)
      # class_weight_per_sample[neutral_mask] *= 0.65

        # 基本 CrossEntropy: -y * log(p)
        per_sample_loss = -(y_onehot * log_probs).sum(dim=1)
        
        # 乘上 class weight
        per_sample_loss = per_sample_loss * class_weight_per_sample

        # 來源加權（dict 較低權重）
        sample_weights = torch.tensor([
            1.0 if s == 'annotated' else 0.3 for s in sources
        ], dtype=torch.float, device=per_sample_loss.device)

        # 最終加權平均 loss
        loss = (per_sample_loss * sample_weights).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y_class).sum().item()
        total += y_class.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(y_class.cpu().numpy())

        acc = correct / total
        loop.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Acc': f"{acc*100:.2f}%"
        })

    avg_loss = total_loss / len(train_loader)
    precision = precision_score(all_trues, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_trues, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)

    train_loss_list.append(avg_loss)
    train_acc_list.append(acc)

    return avg_loss, acc, precision, recall, f1

def evaluate(model, test_loader, label_names, device, show_report=False, show_confusion=True, show_mistakes=False):
    model.eval()
    all_preds = []
    all_trues = []
    all_probs = []
    all_tokens = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                x = batch['input_ids'].to(device)
                y_class = batch['label'].to(device)
                emotion_tensor = batch['emotion_feat'].to(device)
                tokens = batch.get('tokens', ["<N/A>"] * x.size(0))
            else:
                if len(batch) == 3:
                    x, y_class, emotion_tensor = batch
                    x, emotion_tensor = x.to(device), emotion_tensor.to(device)
                    tokens = ["<N/A>"] * x.size(0)
                else:
                    x, y_class = batch
                    emotion_tensor = None
                    x = x.to(device)
                    tokens = ["<N/A>"] * x.size(0)
                y_class = y_class.to(device)

            pred_logits = model(x, emotion_tensor)
            probs = torch.softmax(pred_logits, dim=1)

            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_trues.extend(y_class.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_tokens.extend(tokens)
    
    report = classification_report(all_trues, all_preds, target_names=label_names, zero_division=0, output_dict=True)
    
    if show_report:
       print("\n[Classification Report]")
       print(classification_report(all_trues, all_preds, target_names=label_names, zero_division=0))

    print("\n[Classification Metrics (Macro Average)]")
    precision = precision_score(all_trues, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_trues, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # 儲存每類別 F1（需全域 f1_per_class 已定義）
    for label in ['negative', 'neutral', 'positive']:
        f1_per_class[label].append(report[label]['f1-score'])

    if show_confusion:
        try:
            cm = confusion_matrix(all_trues, all_preds)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Not show Confusion Matrix:{e}")

    if show_mistakes:
        print("\n[Misclassified Samples Analysis]")
        count = 0
        for true, pred, prob, tok in zip(all_trues, all_preds, all_probs, all_tokens):
            if true != pred:
                confidence = prob[pred]
                print(f"→ Predict: {label_names[pred]} | True: {label_names[true]} | Confidence: {confidence:.2f}")
                print("  Tokens :", tok if isinstance(tok, str) else ' '.join(tok))
                print("-" * 60)
                count += 1
                if count >= 30:
                    print("...only showing first 30 mistakes.")
                    break

        # 額外錯誤類別分析（所有錯誤類別對）
        print("\n[Detailed Error Patterns]")
        error_limit = 5
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                if i == j:
                    continue
                print(f"\n→ Misclassified: {label_names[i]} → {label_names[j]}")
                count = 0
                for true, pred, tok in zip(all_trues, all_preds, all_tokens):
                    if true == i and pred == j:
                        print("  Tokens:", tok if isinstance(tok, str) else ' '.join(tok))
                        count += 1
                        if count >= error_limit:
                            print("  ...more skipped.")
                            break

    return f1, report

def train_model(model, train_loader, val_loader, test_loader,
                optimizer, scheduler, device, label_names, tokenizer,
                num_epochs=50):

    if optimizer is None or scheduler is None:
        raise ValueError("Please provide both optimizer and scheduler externally.")

    y_all = torch.tensor([item['label'] for item in train_loader.dataset])
    unique_labels = torch.unique(y_all).cpu().numpy()
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=y_all.numpy())
  # weights[1] *= 0.65
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    best_f1 = 0
    patience = 15
    wait = 0

    # 初始化每類別 F1-score 記錄
    global f1_per_class
    f1_per_class = {label: [] for label in label_names}

    for epoch in range(1, num_epochs + 1):
        avg_loss, acc, precision, recall, f1 = train_one_epoch(
            model, train_loader, class_weights_tensor, optimizer, device, epoch
        )

        print(f"\n[Epoch {epoch}] Average Loss: {avg_loss:.4f}")
        print(f"  → Accuracy : {acc*100:.2f}%")
        print(f"  → Precision: {precision:.4f}")
        print(f"  → Recall   : {recall:.4f}")
        print(f"  → F1 Score : {f1:.4f}")

        print("\n[Validation]")
        val_f1, report = evaluate(model, val_loader, label_names, device,
                                  show_report=(epoch % 5 == 0), show_confusion=False, show_mistakes=False)

        # 每類別 F1-score 更新
        for label in label_names:
            f1_per_class[label].append(report[label]['f1-score'])

        # 總體訓練過程指標紀錄
        train_loss_list.append(avg_loss)
        train_acc_list.append(acc)
        train_precision_list.append(precision)
        train_recall_list.append(recall)
        train_f1_list.append(f1)

        scheduler.step(val_f1)

        delta = 0.001
        if val_f1 > best_f1 + delta:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), "/content/Social_IOT-NLP情緒分析/best_model.pth")
            with open("tokenizer.pickle", "wb") as f:
                pickle.dump(tokenizer, f)
            print("Best model and tokenizer saved.")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered. Training stopped at epoch {epoch}.")
                break

    print("\n=== Final Evaluation on Test Set ===")
    model.load_state_dict(torch.load("/content/Social_IOT-NLP情緒分析/best_model.pth"))
    print("Best model weights loaded.")
    if epoch % 5 == 0:
      evaluate(model, test_loader, label_names, device)

    return model, train_loss_list, train_acc_list, train_precision_list, train_recall_list, train_f1_list, f1_per_class