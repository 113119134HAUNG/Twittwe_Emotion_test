# train.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import os

# Global stats for visualization
train_loss_list, train_acc_list = [], []
train_precision_list, train_recall_list, train_f1_list = [], [], []

def train_one_epoch(model, train_loader, class_weights_tensor, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_trues = [], []
    smoothed_loss = None

    loop = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)
    for batch in loop:
        x = batch['input_ids'].to(device)
        y_class = batch['label'].to(device)
        emotion_tensor = batch['emotion_feat'].to(device)
        sources = batch.get('source', ['annotated'] * x.size(0))  # Default fallback

        optimizer.zero_grad()
        logits = model(x, emotion_tensor)
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        y_onehot = torch.nn.functional.one_hot(y_class, num_classes=logits.size(1)).float()

        class_weight_per_sample = class_weights_tensor[y_class].clone()
        per_sample_loss = -(y_onehot * log_probs).sum(dim=1)
        per_sample_loss *= class_weight_per_sample

        sample_weights = torch.tensor(
            [1.0 if s == 'annotated' else 0.3 for s in sources], device=x.device)
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
        smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss.item() if smoothed_loss else loss.item()
        loop.set_postfix({'Loss': f"{smoothed_loss:.4f}", 'Acc': f"{acc*100:.2f}%"})

    avg_loss = total_loss / len(train_loader)
    precision = precision_score(all_trues, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_trues, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_trues, all_preds, average='macro', zero_division=0)

    train_loss_list.append(avg_loss)
    train_acc_list.append(acc)

    return avg_loss, acc, precision, recall, f1


def evaluate(model, test_loader, label_names, device, show_report=False, show_confusion=True, show_mistakes=False):
    model.eval()
    all_preds, all_trues, all_probs, all_tokens = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['input_ids'].to(device)
            y_class = batch['label'].to(device)
            emotion_tensor = batch['emotion_feat'].to(device)
            tokens = batch.get('tokens', ["<N/A>"] * x.size(0))

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
    print(f"Precision: {precision:.4f}\nRecall   : {recall:.4f}\nF1 Score : {f1:.4f}")

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
            print(f"Unable to plot confusion matrix: {e}")

    return f1, report


def train_model(
    model, train_loader, val_loader, test_loader,
    optimizer, scheduler, device, label_names, tokenizer,
    num_epochs=50,
    model_path="./best_model.pth",
    tokenizer_path="./tokenizer.pickle"
):
    if optimizer is None or scheduler is None:
        raise ValueError("Please provide both optimizer and scheduler externally.")

    y_all = torch.tensor([item['label'] for item in train_loader.dataset])
    unique_labels = torch.unique(y_all).cpu().numpy()
    weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=y_all.numpy())
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

    best_f1 = 0
    patience, wait = 15, 0
    delta = 0.005  # 0.5% 相對改善
    f1_per_class = {label: [] for label in label_names}

    for epoch in range(1, num_epochs + 1):
        avg_loss, acc, precision, recall, f1 = train_one_epoch(
            model, train_loader, class_weights_tensor, optimizer, device, epoch)

        print(f"\n[Epoch {epoch}] Average Loss: {avg_loss:.4f}\n  → Accuracy : {acc*100:.2f}%\n  → Precision: {precision:.4f}\n  → Recall   : {recall:.4f}\n  → F1 Score : {f1:.4f}")

        if len(val_loader) == 0:
            print("Validation loader is empty. Skipping validation and scheduler.")
            continue

        print("\n[Validation]")
        val_f1, report = evaluate(model, val_loader, label_names, device, show_report=(epoch % 5 == 0), show_confusion=False)

        for label in label_names:
            f1_per_class[label].append(report[label]['f1-score'])

        train_precision_list.append(precision)
        train_recall_list.append(recall)
        train_f1_list.append(f1)

        scheduler.step(val_f1)

        # 相對改善 early stopping
        relative_improvement = (val_f1 - best_f1) / (best_f1 + 1e-8)
        if relative_improvement > delta:
            best_f1, wait = val_f1, 0
            torch.save(model.state_dict(), model_path)
            with open(tokenizer_path, "wb") as f:
                pickle.dump(tokenizer, f)
            print(f"Best model and tokenizer saved to:\n- {model_path}\n- {tokenizer_path}")
        else:
            wait += 1
            print(f"No improvement. Wait counter: {wait}/{patience}")
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print("\n=== Final Evaluation on Test Set ===")
    model.load_state_dict(torch.load(model_path))
    print("Best model weights loaded.")
    if epoch % 5 == 0:
        evaluate(model, test_loader, label_names, device)

    return model, train_loss_list, train_acc_list, train_precision_list, train_recall_list, train_f1_list, f1_per_class
