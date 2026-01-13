import os
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

try:
    from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def is_yesno(ans: str) -> bool:
    a = norm_text(ans)
    return a in ["yes", "no"]


def ans_to_label(ans: str) -> int:
    return 1 if norm_text(ans) == "yes" else 0


class SimpleVocab:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def build(self, texts: List[str]):
        freq = {}
        for t in texts:
            for w in norm_text(t).split():
                freq[w] = freq.get(w, 0) + 1
        for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if c >= self.min_freq and w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = []
        for w in norm_text(text).split():
            ids.append(self.stoi.get(w, 1))
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids


class VQARADYesNoDataset(Dataset):
    """
    只取 answer ∈ {yes,no} 的样本
    输入：image + question
    输出：label(0/1)
    """
    def __init__(self, json_path: Path, image_dir: Path, vocab: SimpleVocab,
                 img_tfms, max_q_len=32):
        items = json.load(open(json_path, "r", encoding="utf-8"))
        if isinstance(items, dict) and "data" in items:
            items = items["data"]

        self.samples = []
        for i, ex in enumerate(items):
            img = ex.get("image_name") or ex.get("image")
            q = ex.get("question")
            a = ex.get("answer")

            if isinstance(a, list) and len(a) > 0:
                a = a[0]

            if img is None or q is None or a is None:
                continue
            if not is_yesno(a):
                continue

            self.samples.append({
                "qid": str(ex.get("qid") or ex.get("question_id") or i),
                "image_name": str(img),
                "question": str(q),
                "answer": str(a),
                "label": ans_to_label(a)
            })

        self.image_dir = image_dir
        self.vocab = vocab
        self.img_tfms = img_tfms
        self.max_q_len = max_q_len

        if len(self.samples) == 0:
            raise RuntimeError(f"No yes/no samples found in {json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self.image_dir / s["image_name"]
        image = Image.open(img_path).convert("RGB")
        image = self.img_tfms(image)

        q_ids = torch.tensor(self.vocab.encode(s["question"], self.max_q_len), dtype=torch.long)
        y = torch.tensor(s["label"], dtype=torch.long)

        meta = {
            "qid": s["qid"],
            "image_name": s["image_name"],
            "question": s["question"],
            "answer": s["answer"]
        }
        return image, q_ids, y, meta


class CNNTextYesNo(nn.Module):
    def __init__(self, vocab_size: int, txt_emb=128, txt_hidden=128, img_feat=256):
        super().__init__()

        # Image encoder: ResNet18 -> 256 dim
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.img_proj = nn.Linear(in_feat, img_feat)

        # Text encoder: Embedding + GRU -> 128 dim
        self.emb = nn.Embedding(vocab_size, txt_emb, padding_idx=0)
        self.gru = nn.GRU(input_size=txt_emb, hidden_size=txt_hidden,
                          batch_first=True, bidirectional=False)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(img_feat + txt_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, images, q_ids):
        img_vec = self.backbone(images)                 # [B, 512]
        img_vec = self.img_proj(img_vec)                # [B, 256]

        x = self.emb(q_ids)                             # [B, T, E]
        _, h = self.gru(x)                              # h: [1, B, H]
        txt_vec = h.squeeze(0)                          # [B, H]

        feat = torch.cat([img_vec, txt_vec], dim=1)
        logits = self.classifier(feat)
        return logits


def compute_metrics(y_true, y_prob):
    """
    y_prob: probability of class 1 (yes)
    """
    y_pred = (y_prob >= 0.5).astype(int)
    if SKLEARN_OK:
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float("nan")
    else:
        acc = float((y_pred == y_true).mean())
        # simple recall/f1 fallback
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        auc = float("nan")

    return acc, rec, f1, auc


@torch.no_grad()
def run_eval(model, loader, device, criterion):
    model.eval()
    losses = []
    y_true = []
    y_prob = []

    for images, q_ids, y, _ in loader:
        images = images.to(device)
        q_ids = q_ids.to(device)
        y = y.to(device)

        logits = model(images, q_ids)
        loss = criterion(logits, y)
        losses.append(float(loss.item()))

        prob1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        y_prob.append(prob1)
        y_true.append(y.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    acc, rec, f1, auc = compute_metrics(y_true, y_prob)
    return float(np.mean(losses)), acc, rec, f1, auc


def plot_curve(xs, tr, va, title, ylabel, out_path: Path):
    plt.figure()
    plt.plot(xs, tr, label="Train")
    plt.plot(xs, va, label="Validation")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main(
    epochs=10,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    max_q_len=32,
    seed=42
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    data_dir = Path("data/vqa-rad")
    image_dir = data_dir / "VQA_RAD Image Folder"

    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"
    test_json = data_dir / "test.json"

    out_dir = Path("outputs/cnn_yesno")
    out_dir.mkdir(parents=True, exist_ok=True)

    img_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Build vocab from YES/NO train questions only (fair)
    raw_train = json.load(open(train_json, "r", encoding="utf-8"))
    if isinstance(raw_train, dict) and "data" in raw_train:
        raw_train = raw_train["data"]
    train_questions = []
    for ex in raw_train:
        a = ex.get("answer")
        if isinstance(a, list) and len(a) > 0:
            a = a[0]
        if a is None or ex.get("question") is None:
            continue
        if is_yesno(a):
            train_questions.append(ex["question"])

    vocab = SimpleVocab(min_freq=1)
    vocab.build(train_questions)
    print("Vocab size:", len(vocab.itos))

    train_ds = VQARADYesNoDataset(train_json, image_dir, vocab, img_tfms, max_q_len=max_q_len)
    val_ds = VQARADYesNoDataset(val_json, image_dir, vocab, img_tfms, max_q_len=max_q_len)
    test_ds = VQARADYesNoDataset(test_json, image_dir, vocab, img_tfms, max_q_len=max_q_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CNNTextYesNo(vocab_size=len(vocab.itos)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_rows = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        y_true = []
        y_prob = []

        pbar = tqdm(train_loader, desc=f"train ep{ep}")
        for images, q_ids, y, _ in pbar:
            images = images.to(device)
            q_ids = q_ids.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(images, q_ids)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            tr_losses.append(float(loss.item()))
            prob1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            y_prob.append(prob1)
            y_true.append(y.detach().cpu().numpy())
            pbar.set_postfix(loss=float(loss.item()))

        y_true_np = np.concatenate(y_true)
        y_prob_np = np.concatenate(y_prob)
        tr_loss = float(np.mean(tr_losses))
        tr_acc, tr_rec, tr_f1, tr_auc = compute_metrics(y_true_np, y_prob_np)

        va_loss, va_acc, va_rec, va_f1, va_auc = run_eval(model, val_loader, device, criterion)

        print(
            f"Epoch {ep:02d} | "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | "
            f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} | "
            f"train_f1={tr_f1:.4f} val_f1={va_f1:.4f}"
        )

        log_rows.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_accuracy": tr_acc,
            "val_accuracy": va_acc,
            "train_recall": tr_rec,
            "val_recall": va_rec,
            "train_f1": tr_f1,
            "val_f1": va_f1,
            "train_roc_auc": tr_auc,
            "val_roc_auc": va_auc,
        })

    df = pd.DataFrame(log_rows)
    df.to_csv(out_dir / "train_log.csv", index=False, encoding="utf-8-sig")

    xs = df["epoch"].tolist()
    plot_curve(xs, df["train_accuracy"], df["val_accuracy"], "CNN Accuracy Over Epochs", "Accuracy", out_dir / "cnn_accuracy.png")
    plot_curve(xs, df["train_loss"], df["val_loss"], "CNN Loss Over Epochs", "Loss", out_dir / "cnn_loss.png")
    plot_curve(xs, df["train_recall"], df["val_recall"], "CNN Recall Over Epochs", "Recall", out_dir / "cnn_recall.png")
    plot_curve(xs, df["train_f1"], df["val_f1"], "CNN F1 Score Over Epochs", "F1", out_dir / "cnn_f1.png")
    plot_curve(xs, df["train_roc_auc"], df["val_roc_auc"], "CNN ROC-AUC Over Epochs", "ROC-AUC", out_dir / "cnn_roc_auc.png")

    # Final test metrics
    te_loss, te_acc, te_rec, te_f1, te_auc = run_eval(model, test_loader, device, criterion)
    print("\n=== CNN YES/NO Test Metrics ===")
    print(f"Loss: {te_loss:.4f}")
    print(f"Accuracy: {te_acc:.4f}")
    print(f"Recall: {te_rec:.4f}")
    print(f"F1: {te_f1:.4f}")
    print(f"ROC-AUC: {te_auc:.4f}")


if __name__ == "__main__":
    main(
        epochs=10,
        batch_size=16,
        lr=1e-4,
        weight_decay=1e-4,
        max_q_len=32,
        seed=42
    )
