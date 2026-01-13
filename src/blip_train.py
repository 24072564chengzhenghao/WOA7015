# src/blip_train.py
import os
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import BlipProcessor, BlipForQuestionAnswering


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    # simple normalize
    for ch in ["\n", "\t", "\r"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gt: str) -> int:
    return int(normalize_text(pred) == normalize_text(gt))


def is_yesno_answer(ans: str) -> bool:
    a = normalize_text(ans)
    return a in ["yes", "no"]


@dataclass
class ItemMeta:
    qid: str
    image_name: str
    question: str
    answer: str


# -------------------------
# Dataset
# -------------------------
class VQARADJsonDataset(Dataset):
    """
    Expects VQA-RAD json format in your starter:
      items = json.load(open("train.json"))
      each item has at least: image_name (or image), question, answer
    """
    def __init__(self, json_path: Path, image_dir: Path):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)

        items = json.load(open(self.json_path, "r", encoding="utf-8"))
        # Some VQA-RAD json variants: list[dict] or dict with "data"
        if isinstance(items, dict) and "data" in items:
            items = items["data"]

        self.items: List[ItemMeta] = []
        for i, ex in enumerate(items):
            img = ex.get("image_name") or ex.get("image") or ex.get("img_id") or ex.get("image_id")
            q = ex.get("question") or ex.get("query")
            a = ex.get("answer") or ex.get("label") or ex.get("answers")

            # answers might be list
            if isinstance(a, list) and len(a) > 0:
                a = a[0]

            if img is None or q is None or a is None:
                # skip bad rows
                continue

            qid = str(ex.get("qid") or ex.get("question_id") or i)
            self.items.append(ItemMeta(qid=qid, image_name=str(img), question=str(q), answer=str(a)))

        if len(self.items) == 0:
            raise RuntimeError(f"No valid items parsed from: {self.json_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        m = self.items[idx]
        img_path = self.image_dir / m.image_name
        if not img_path.exists():
            # sometimes images stored without extension changes; try raw name
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        return {
            "qid": m.qid,
            "image": image,           # PIL Image (CPU)
            "question": m.question,   # str
            "answer": m.answer,       # str
            "image_name": m.image_name
        }


# -------------------------
# Collate
# -------------------------
def make_collate_fn(processor: BlipProcessor, max_q_len: int = 64, max_a_len: int = 16):
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str], List[Dict[str, Any]]]:
        images = [b["image"] for b in batch]
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]

        # question + image -> model inputs
        enc = processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_q_len
        )

        # answer tokens -> labels
        ans_tok = processor.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_a_len
        )
        labels = ans_tok["input_ids"]

        # pad -> -100 (ignore_index) for loss
        labels = labels.clone()
        labels[labels == pad_id] = -100

        metas = [{
            "qid": b["qid"],
            "image_name": b["image_name"],
            "question": b["question"],
            "answer": b["answer"],
            "is_yesno": is_yesno_answer(b["answer"])
        } for b in batch]

        return enc, labels, answers, metas

    return collate


# -------------------------
# Eval helpers
# -------------------------
@torch.no_grad()
def evaluate(model, processor, loader, device, max_new_tokens: int = 8):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    em_hits = 0
    em_total = 0

    yn_hits = 0
    yn_total = 0

    unk_count = 0

    for enc, labels, answers, metas in tqdm(loader, desc="eval", leave=False):
        # move to device here (IMPORTANT: collate returns CPU tensors)
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)

        out = model(**enc, labels=labels)
        loss = out.loss
        total_loss += float(loss.item())
        n_batches += 1

        # generate predictions
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
        )
        preds = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for p, m in zip(preds, metas):
            gt = m["answer"]
            em_hits += exact_match(p, gt)
            em_total += 1

            if m["is_yesno"]:
                yn_total += 1
                yn_hits += exact_match(p, gt)

            if normalize_text(p) in ["", "unknown", "unk", "<unk>"]:
                unk_count += 1

    avg_loss = total_loss / max(1, n_batches)
    em = em_hits / max(1, em_total)
    yn_acc = yn_hits / max(1, yn_total) if yn_total > 0 else 0.0
    unk_rate = unk_count / max(1, em_total)
    return avg_loss, em, yn_acc, unk_rate


@torch.no_grad()
def predict_to_csv(model, processor, loader, device, out_csv: Path, max_new_tokens: int = 8):
    model.eval()
    rows = []
    for enc, labels, answers, metas in tqdm(loader, desc="test", leave=False):
        enc = {k: v.to(device) for k, v in enc.items()}
        gen_ids = model.generate(**enc, max_new_tokens=max_new_tokens)
        preds = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for p, m in zip(preds, metas):
            rows.append({
                "qid": m["qid"],
                "image_name": m["image_name"],
                "question": m["question"],
                "gt_answer": m["answer"],
                "pred_answer": p,
                "is_yesno": int(m["is_yesno"]),
                "em": exact_match(p, m["answer"])
            })

    import pandas as pd
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


# -------------------------
# Train
# -------------------------
def main(
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 2e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 0,
    seed: int = 42,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    data_dir = Path("data/vqa-rad")
    image_dir = data_dir / "VQA_RAD Image Folder"

    train_path = data_dir / "train.json"
    val_path = data_dir / "val.json"
    test_path = data_dir / "test.json"

    out_dir = Path("outputs/blip_finetune")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)

    print("Tokenizer vocab:", len(processor.tokenizer))

    # Data
    train_ds = VQARADJsonDataset(train_path, image_dir)
    val_ds = VQARADJsonDataset(val_path, image_dir)
    test_ds = VQARADJsonDataset(test_path, image_dir)

    collate_fn = make_collate_fn(processor)

    # IMPORTANT:
    # - keep num_workers=0 on Windows for stability
    # - pin_memory=False because we move tensors to GPU in the loop (and we keep CPU tensors from collate)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False, collate_fn=collate_fn)

    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Logging
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,train_em,val_em,train_yesno_acc,val_yesno_acc,val_unk_rate\n")

    best_val_em = -1.0
    best_path = out_dir / "best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        # For train EM/YN acc, we do a lightweight eval using generate on a small subset to save time
        # (full train generate every epoch is slow)
        train_preds = []
        train_gts = []
        train_is_yesno = []

        pbar = tqdm(train_loader, desc=f"train ep{ep}", leave=False)
        for enc, labels, answers, metas in pbar:
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(**enc, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=float(loss.item()))

            # collect a few samples for train EM estimate
            if len(train_preds) < 200:
                with torch.no_grad():
                    gen_ids = model.generate(**enc, max_new_tokens=8)
                    preds = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                for p, m in zip(preds, metas):
                    train_preds.append(p)
                    train_gts.append(m["answer"])
                    train_is_yesno.append(m["is_yesno"])

        train_loss = total_loss / max(1, steps)

        # train EM estimate
        train_em = sum(exact_match(p, g) for p, g in zip(train_preds, train_gts)) / max(1, len(train_gts))
        yn_hits = 0
        yn_total = 0
        for p, g, isyn in zip(train_preds, train_gts, train_is_yesno):
            if isyn:
                yn_total += 1
                yn_hits += exact_match(p, g)
        train_yesno_acc = yn_hits / yn_total if yn_total > 0 else 0.0

        # val full eval
        val_loss, val_em, val_yesno_acc, val_unk_rate = evaluate(model, processor, val_loader, device)

        line = (
            f"Epoch {ep:02d}: "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_em={train_em:.4f} | val_em={val_em:.4f} | "
            f"train_yesno_acc={train_yesno_acc:.4f} | val_yesno_acc={val_yesno_acc:.4f} | "
            f"val_unk_rate={val_unk_rate:.4f}"
        )
        print(line)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ep},{train_loss:.6f},{val_loss:.6f},{train_em:.6f},{val_em:.6f},{train_yesno_acc:.6f},{val_yesno_acc:.6f},{val_unk_rate:.6f}\n")

        # save best
        if val_em > best_val_em:
            best_val_em = val_em
            torch.save({"model": model.state_dict()}, best_path)
            print("saved best.pt")

    # load best and test
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

    out_csv = out_dir / "blip_test_predictions.csv"
    predict_to_csv(model, processor, test_loader, device, out_csv)
    print(f"[OK] Saved: {out_csv}")
    print(f"[OK] Train log: {log_path}")
    print(f"[OK] Best val EM: {best_val_em:.4f}")


if __name__ == "__main__":
    # RTX 4060 8GB: batch_size=2 is usually OK; if OOM -> 1
    main(
        epochs=10,
        batch_size=2,
        lr=2e-5,
        weight_decay=1e-2,
        num_workers=0,
        seed=42,
    )
