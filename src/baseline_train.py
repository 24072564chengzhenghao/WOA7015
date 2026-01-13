# src/baseline_train.py
import os
import re
import csv
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

from transformers import AutoTokenizer, AutoModel


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_str(x) -> str:
    """Convert any answer to a safe string."""
    if x is None:
        return ""
    # some datasets store answers as int/float/bool
    try:
        return str(x)
    except Exception:
        return ""


def normalize_text(s: str) -> str:
    """Simple normalization for EM (exact match)."""
    s = safe_str(s).lower().strip()
    # remove extra spaces
    s = re.sub(r"\s+", " ", s)
    # remove punctuation (keep slash because medical answers sometimes use it)
    s = re.sub(r"[^\w\s/]", "", s)
    return s


def build_answer_vocab(items, topk: int = 500):
    """
    Build answer vocabulary from training items.
    vocab[0] is <unk>.
    """
    counter = Counter()
    for x in items:
        a = normalize_text(x.get("answer", ""))
        if a != "":
            counter[a] += 1

    most_common = counter.most_common(topk)
    vocab = ["<unk>"] + [a for a, _ in most_common]
    stoi = {a: i for i, a in enumerate(vocab)}
    return vocab, stoi


def compute_em(preds_str, gts_str):
    """Exact match score."""
    hit = 0
    for p, g in zip(preds_str, gts_str):
        if normalize_text(p) == normalize_text(g):
            hit += 1
    return hit / max(1, len(preds_str))


def compute_yesno_acc(preds_str, gts_str):
    """
    Accuracy on yes/no questions only (based on GT being yes/no).
    """
    labels = {"yes", "no"}
    total = 0
    hit = 0
    for p, g in zip(preds_str, gts_str):
        ng = normalize_text(g)
        if ng in labels:
            total += 1
            if normalize_text(p) == ng:
                hit += 1
    if total == 0:
        return 0.0
    return hit / total


# -------------------------
# Dataset
# -------------------------
class VQARADDataset(Dataset):
    """
    Expects each item to have at least:
      - question (or "query")
      - answer (may be str/int/None)
      - image filename key: one of ["image_name","image","img","image_path","image_id"]
    """

    def __init__(self, json_path: str | Path, image_dir: str | Path, tokenizer, transform=None, max_len: int = 32):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.items = json.load(open(self.json_path, "r", encoding="utf-8"))
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def _get_image_filename(self, x: dict) -> str:
        for k in ["image_name", "image", "img", "image_path", "image_id"]:
            if k in x and x[k] is not None:
                return safe_str(x[k])
        # fallback
        return safe_str(x.get("id", ""))

    def _get_question(self, x: dict) -> str:
        if "question" in x and x["question"] is not None:
            return safe_str(x["question"])
        if "query" in x and x["query"] is not None:
            return safe_str(x["query"])
        return ""

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        x = self.items[idx]

        q = self._get_question(x)
        a = safe_str(x.get("answer", ""))

        img_name = self._get_image_filename(x)

        # If json stored full path, use it directly; else join with image_dir
        img_path = Path(img_name)
        if not img_path.exists():
            img_path = self.image_dir / img_name

        # Some files may miss extension; try common ones
        if not img_path.exists():
            for ext in [".jpg", ".png", ".jpeg"]:
                if (self.image_dir / (img_name + ext)).exists():
                    img_path = self.image_dir / (img_name + ext)
                    break

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        toks = self.tokenizer(
            q,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        toks = {k: v.squeeze(0) for k, v in toks.items()}  # [L]

        meta = {
            "image": img_name,
            "question": q,
        }
        return image, toks, a, meta


def collate_fn(batch):
    imgs, toks, answers, metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    input_ids = torch.stack([t["input_ids"] for t in toks], dim=0)
    attention_mask = torch.stack([t["attention_mask"] for t in toks], dim=0)

    token_type_ids = None
    if "token_type_ids" in toks[0]:
        token_type_ids = torch.stack([t["token_type_ids"] for t in toks], dim=0)

    return imgs, input_ids, attention_mask, token_type_ids, list(answers), list(metas)


# -------------------------
# Model
# -------------------------
class BaselineVQA(nn.Module):
    """
    Simple baseline:
      - DenseNet121 image encoder (pretrained)
      - BERT text encoder (bert-base-uncased)
      - concat + classifier
    """

    def __init__(self, num_answers: int, text_model_name: str = "bert-base-uncased", img_feat_dim: int = 512, txt_feat_dim: int = 512):
        super().__init__()

        # Image encoder
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.img_backbone = densenet.features
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, img_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Text encoder
        self.txt_backbone = AutoModel.from_pretrained(text_model_name)
        hidden = self.txt_backbone.config.hidden_size
        self.txt_proj = nn.Sequential(
            nn.Linear(hidden, txt_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Fusion head
        self.classifier = nn.Sequential(
            nn.Linear(img_feat_dim + txt_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_answers),
        )

    def forward(self, images, input_ids, attention_mask, token_type_ids=None):
        # image
        x = self.img_backbone(images)
        x = self.img_pool(x)
        img_feat = self.img_proj(x)

        # text
        if token_type_ids is not None:
            out = self.txt_backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            out = self.txt_backbone(input_ids=input_ids, attention_mask=attention_mask)

        # use CLS token
        cls_feat = out.last_hidden_state[:, 0, :]  # [B, H]
        txt_feat = self.txt_proj(cls_feat)

        # fuse
        feat = torch.cat([img_feat, txt_feat], dim=1)
        logits = self.classifier(feat)
        return logits


# -------------------------
# Train / Eval loops
# -------------------------
@torch.no_grad()
def run_eval(model, loader, device, vocab):
    model.eval()

    all_preds = []
    all_gts = []
    unk_count = 0
    total = 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()

    for imgs, input_ids, attn, token_type_ids, answers, metas in loader:
        imgs = imgs.to(device)
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = model(imgs, input_ids, attn, token_type_ids)
        pred_idx = logits.argmax(dim=1).detach().cpu().tolist()

        # For loss, need labels. We map gt string->index; unknown -> 0
        # BUT we don't have stoi here, so compute loss outside OR skip.
        # We'll compute loss as "None-safe" by rebuilding a small stoi from vocab:
        stoi = {a: i for i, a in enumerate(vocab)}
        labels = []
        for a in answers:
            na = normalize_text(a)
            labels.append(stoi.get(na, 0))
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        loss = ce(logits, labels)
        total_loss += float(loss.item()) * len(answers)

        for i in pred_idx:
            if i == 0:
                unk_count += 1
            all_preds.append(vocab[i] if i < len(vocab) else "<unk>")
        all_gts.extend([safe_str(a) for a in answers])

        total += len(answers)

    em = compute_em(all_preds, all_gts)
    yesno_acc = compute_yesno_acc(all_preds, all_gts)
    unk_rate = unk_count / max(1, total)
    avg_loss = total_loss / max(1, total)
    return avg_loss, em, yesno_acc, unk_rate


def train_one_epoch(model, loader, device, optimizer, ce, vocab, stoi):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_gts = []

    for imgs, input_ids, attn, token_type_ids, answers, metas in loader:
        imgs = imgs.to(device)
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # labels
        labels = []
        for a in answers:
            na = normalize_text(a)
            labels.append(stoi.get(na, 0))
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs, input_ids, attn, token_type_ids)
        loss = ce(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(answers)

        pred_idx = logits.argmax(dim=1).detach().cpu().tolist()
        for i in pred_idx:
            all_preds.append(vocab[i] if i < len(vocab) else "<unk>")
        all_gts.extend([safe_str(a) for a in answers])

    avg_loss = total_loss / max(1, len(loader.dataset))
    em = compute_em(all_preds, all_gts)
    yesno_acc = compute_yesno_acc(all_preds, all_gts)
    return avg_loss, em, yesno_acc


# -------------------------
# Main
# -------------------------
def main(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    topk: int = 500,
    max_len: int = 32,
    seed: int = 42,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    # Paths
    data_dir = Path("data/vqa-rad")
    image_dir = data_dir / "VQA_RAD Image Folder"
    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"

    out_dir = Path("outputs/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Transforms
    tfm = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load items for vocab
    train_items = json.load(open(train_path, "r", encoding="utf-8"))
    vocab, stoi = build_answer_vocab(train_items, topk=topk)
    print(f"Vocab size: {len(vocab)} (topk={topk}, + <unk>)")

    # Datasets / loaders
    train_ds = VQARADDataset(train_path, image_dir, tokenizer=tokenizer, transform=tfm, max_len=max_len)
    val_ds = VQARADDataset(val_path, image_dir, tokenizer=tokenizer, transform=tfm, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Model
    model = BaselineVQA(num_answers=len(vocab)).to(device)

    # Optim / loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    # Log CSV (the columns plot_results.py expects)
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "train_em",
                "val_em",
                "train_yesno_acc",
                "val_yesno_acc",
                "val_unk_rate",
            ]
        )

    best_em = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_em, train_yesno_acc = train_one_epoch(model, train_loader, device, optimizer, ce, vocab, stoi)
        val_loss, val_em, val_yesno_acc, val_unk_rate = run_eval(model, val_loader, device, vocab)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_em={train_em:.4f} | val_em={val_em:.4f} | "
            f"train_yesno_acc={train_yesno_acc:.4f} | val_yesno_acc={val_yesno_acc:.4f} | "
            f"val_unk_rate={val_unk_rate:.4f}"
        )

        # append log
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{train_em:.6f}",
                    f"{val_em:.6f}",
                    f"{train_yesno_acc:.6f}",
                    f"{val_yesno_acc:.6f}",
                    f"{val_unk_rate:.6f}",
                ]
            )

        # save best
        if val_em > best_em:
            best_em = val_em
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "topk": topk,
                        "max_len": max_len,
                        "seed": seed,
                    },
                },
                out_dir / "best.pt",
            )
            print("  saved best.pt")

    print("Best val EM:", best_em)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    # Keep defaults aligned with your current runs
    main(
        epochs=10,
        batch_size=16,
        lr=1e-4,
        weight_decay=1e-2,
        topk=500,
        max_len=32,
        seed=42,
    )
