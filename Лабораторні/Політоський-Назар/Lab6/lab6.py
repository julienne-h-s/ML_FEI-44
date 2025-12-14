#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lab6.py
Повне рішення лабораторної №6: RNN vs LSTM для класифікації Fake/Real News (PyTorch)

Запуск приклад:
    python lab6.py --epochs 5 --batch-size 64 --mode demo

Основні опції:
    --data-dir       : директорія з Fake.csv і True.csv (за замовчуванням - поточна)
    --max-len        : максимальна довжина послідовності (паддінг/тримінг)
    --vocab-size     : максимально допустимий розмір словника
    --min-freq       : мінімальна частота токена, щоб він потрапив у словник
    --model          : 'lstm' або 'rnn'
    --bidirectional  : використовувати bidirectional LSTM (тільки для модель=lstm)
    --epochs, --batch-size, --lr : гіперпараметри
    --mode           : 'demo' (швидкий прогін на підвибірці) або 'full' (повний)
    --save-dir       : куди зберігати результати (моделі, словник)
"""

import os
import sys
import argparse
import random
import time
import json
import html
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Константи
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# ---------------------------
# Утиліти: очищення та токенізація
# ---------------------------
def clean_text(text: str) -> str:
    """Очищення тексту: lower, видалення URL та email, лишає базову пунктуацію"""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    # дозволяємо латинські літери, цифри, пробіли та небагато пунктуації
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\;\:\'\"\-()]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str):
    """Простий word-level tokenizer (split по пробілах)"""
    return text.split()

def prepare_text_row(row: pd.Series, use_title: bool = True):
    parts = []
    if use_title and isinstance(row.get('title',''), str):
        parts.append(row.get('title',''))
    if isinstance(row.get('text',''), str):
        parts.append(row.get('text',''))
    text = " ".join(parts)
    text = clean_text(text)
    return tokenize(text)

# ---------------------------
# Побудова словника, кодування, паддінг
# ---------------------------
def build_vocab(token_lists, max_vocab=20000, min_freq=2, reserved_tokens=None):
    reserved_tokens = reserved_tokens or [PAD_TOKEN, UNK_TOKEN]
    counter = Counter(tok for toks in token_lists for tok in toks)
    items = [tok for tok, cnt in counter.most_common() if cnt >= min_freq]
    items = items[: max_vocab - len(reserved_tokens)]
    itos = reserved_tokens + items
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos

def encode_and_pad(token_lists, stoi, max_len=400):
    pad_idx = stoi.get(PAD_TOKEN, 0)
    unk_idx = stoi.get(UNK_TOKEN, 1)
    X = []
    for toks in token_lists:
        ids = [stoi.get(tok, unk_idx) for tok in toks][:max_len]
        if len(ids) < max_len:
            ids = ids + [pad_idx] * (max_len - len(ids))
        X.append(ids)
    return np.array(X, dtype=np.int64)

# ---------------------------
# Dataset
# ---------------------------
class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# Моделі: Simple RNN та LSTM
# ---------------------------
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True, nonlinearity='tanh',
                          dropout=(dropout if num_layers > 1 else 0.0))
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)  # (B, L, emb_dim)
        out, h_n = self.rnn(emb)  # h_n: (num_layers, B, H)
        last = h_n[-1]           # (B, H)
        logits = self.fc(last).squeeze(1)
        probs = torch.sigmoid(logits)
        return probs, logits

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128, num_layers=1, bidirectional=False, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=(dropout if num_layers > 1 else 0.0))
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * factor, 1)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h_n, c_n) = self.lstm(emb)
        if self.lstm.bidirectional:
            # last forward + last backward
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]
        logits = self.fc(last).squeeze(1)
        probs = torch.sigmoid(logits)
        return probs, logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------
# Навчання та оцінка
# ---------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, clip=5.0):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        probs, logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    ys, preds, probs = [], [], []
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        p, logits = model(X_batch)
        ys.append(y_batch.cpu().numpy())
        probs.append(p.cpu().numpy())
        preds.append((p.cpu().numpy() >= 0.5).astype(int))
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    acc = accuracy_score(ys, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(ys, preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(ys, probs)
    except Exception:
        auc = float('nan')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
            'y_true': ys, 'y_pred': preds, 'y_prob': probs}

def train_model(model, train_loader, val_loader, device, lr=1e-3, epochs=5, weight_decay=0.0, save_best_path=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_val_f1 = -1.0
    best_state = None
    history = {'train_loss': [], 'val_f1': [], 'val_auc': []}
    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_res = evaluate_model(model, val_loader, device)
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_res['f1'])
        history['val_auc'].append(val_res['auc'])
        dt = time.time() - t0
        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} val_f1={val_res['f1']:.4f} val_auc={val_res['auc']:.4f} time={dt:.1f}s")
        if val_res['f1'] > best_val_f1:
            best_val_f1 = val_res['f1']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # load best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        if save_best_path:
            torch.save(model.state_dict(), save_best_path)
    return model, history

# ---------------------------
# Візуалізації (matplotlib)
# ---------------------------
def plot_history(history, save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='train_loss')
    if 'val_f1' in history:
        plt.plot(history['val_f1'], label='val_f1')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Training history')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_roc(y_true, y_prob, save_path=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_confusion(y_true, y_pred, save_path=None):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, val, ha='center', va='center')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# ---------------------------
# Головна функція: підготовка даних, тренування, оцінка
# ---------------------------
def main(args):
    # пристрій
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    # файли
    data_dir = Path(args.data_dir)
    fake_path = data_dir / "Fake.csv"
    true_path = data_dir / "True.csv"
    if not fake_path.exists() or not true_path.exists():
        print("Помилка: не знайдено Fake.csv або True.csv у директорії:", data_dir)
        print("Поклади потрібні файли у ту ж папку або вкажи --data-dir")
        sys.exit(1)
    # читання
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    df_fake['label'] = 1
    df_true['label'] = 0
    df = pd.concat([df_fake, df_true], ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print("Зчитано. Розмір датасету:", df.shape)
    # токенізація
    print("Токенізація...")
    token_lists = [prepare_text_row(r, use_title=args.use_title) for _, r in df.iterrows()]
    # побудова словника
    print("Побудова словника...")
    stoi, itos = build_vocab(token_lists, max_vocab=args.vocab_size, min_freq=args.min_freq)
    print("Vocab size (final):", len(itos))
    # кодування
    print("Кодування та паддінг...")
    X_all = encode_and_pad(token_lists, stoi, max_len=args.max_len)
    y_all = df['label'].values.astype(np.int64)
    # розбиття
    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp)
    print("Split: train", X_train.shape, "val", X_val.shape, "test", X_test.shape)
    # Якщо demo — використовувати підмножину для швидкого прогону
    if args.mode == "demo":
        max_demo = args.demo_size
        if X_train.shape[0] > max_demo:
            X_train = X_train[:max_demo]
            y_train = y_train[:max_demo]
            print(f"DEMO режим: використано перші {max_demo} зразків для тренування")
    # DataLoaders
    train_loader = DataLoader(NewsDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NewsDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(NewsDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    # модель
    vocab_size = len(itos)
    if args.model == "lstm":
        model = LSTMClassifier(vocab_size=vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                               num_layers=args.num_layers, bidirectional=args.bidirectional, dropout=args.dropout)
    else:
        model = SimpleRNNClassifier(vocab_size=vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                                    num_layers=args.num_layers, dropout=args.dropout)
    print("Model:", args.model, "| params:", count_parameters(model))
    # результати та директорія збереження
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = save_dir / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({"itos": itos, "stoi": stoi}, f, ensure_ascii=False)
    # навчання
    print("Старт навчання...")
    save_model_path = save_dir / f"best_{args.model}.pt"
    trained_model, history = train_model(model, train_loader, val_loader, device=device, lr=args.lr, epochs=args.epochs,
                                         weight_decay=args.weight_decay, save_best_path=save_model_path)
    # оцінка на тесті
    print("Оцінка на тестовому наборі...")
    test_res = evaluate_model(trained_model, test_loader, device)
    print("Test results:")
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        print(f"  {k}: {test_res[k]:.4f}")
    # візуалізації та збереження графіків
    try:
        plot_history(history, save_path=str(save_dir / "history.png"))
        plot_roc(test_res['y_true'], test_res['y_prob'], save_path=str(save_dir / "roc.png"))
        plot_confusion(test_res['y_true'], test_res['y_pred'], save_path=str(save_dir / "confusion.png"))
    except Exception as e:
        print("Warning: не вдалося побудувати або зберегти графіки:", e)
    # збереження фінального звіту JSON
    summary = {
        "model": args.model,
        "params": {
            "vocab_size": vocab_size, "emb_dim": args.emb_dim, "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers, "bidirectional": args.bidirectional, "dropout": args.dropout
        },
        "metrics_test": {k: float(test_res[k]) for k in ("accuracy", "precision", "recall", "f1", "auc")},
        "history": history
    }
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Результати збережено у:", save_dir)

# ---------------------------
# Парсер аргументів
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Lab6: RNN vs LSTM for Fake News (PyTorch).")
    parser.add_argument("--data-dir", type=str, default=".", help="Директорія з Fake.csv та True.csv")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "full"], help="demo - швидкий прогін; full - повний")
    parser.add_argument("--demo-size", type=int, default=4000, help="Розмір підмножини для demo режиму")
    parser.add_argument("--max-len", type=int, default=400, help="Макс. довжина послідовності")
    parser.add_argument("--vocab-size", type=int, default=20000, help="Макс. розмір словника")
    parser.add_argument("--min-freq", type=int, default=2, help="Мінімальна частота токена для включення у словник")
    parser.add_argument("--use-title", action="store_true", help="Використовувати title + text (інакше тільки text)")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm","rnn"], help="lstm або rnn")
    parser.add_argument("--bidirectional", action="store_true", help="Якщо model=lstm, зробити його bidirectional")
    parser.add_argument("--emb-dim", type=int, default=100, help="Розмір embedding")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Розмір hidden")
    parser.add_argument("--num-layers", type=int, default=1, help="Кількість шарів RNN/LSTM")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--epochs", type=int, default=5, help="Кількість епох")
    parser.add_argument("--batch-size", "--batch_size", type=int, dest="batch_size", default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (Adam)")
    parser.add_argument("--save-dir", type=str, default="results_lab6", help="Директорія для збереження результатів")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # run
    main(args)
