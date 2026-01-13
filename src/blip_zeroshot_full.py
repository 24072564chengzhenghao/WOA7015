# src/blip_zeroshot_full.py
import os
import json
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering


def norm_answer(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def is_yesno(ans: str) -> bool:
    a = norm_answer(ans)
    return a in {"yes", "no"}


@torch.no_grad()
def main():
    # ===== Paths =====
    data_dir = Path("data/vqa-rad")
    image_dir = data_dir / "VQA_RAD Image Folder"
    test_path = data_dir / "test.json"

    out_dir = Path("outputs/blip_zeroshot")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "blip_test_predictions.csv"
    out_metrics = out_dir / "blip_metrics.txt"
    out_cases = out_dir / "blip_5_cases.txt"

    # ===== Device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    # ===== Load data =====
    items = json.load(open(test_path, "r", encoding="utf-8"))
    # items: list of dicts, typically contains: image_name, question, answer, qid(optional)
    print("Test size:", len(items))

    # ===== Model =====
    model_name = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
    model.eval()

    rows = []
    for i, it in enumerate(tqdm(items, desc="BLIP zero-shot test")):
        qid = it.get("qid", i)
        img_name = it.get("image_name") or it.get("image") or it.get("img_name")
        question = it.get("question", "")
        gt = it.get("answer") or it.get("gt_answer") or it.get("answer_text") or ""

        img_path = image_dir / img_name
        if not img_path.exists():
            # 兜底：有些json可能存的是相对路径
            img_path = image_dir / Path(str(img_name)).name

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            rows.append({
                "qid": qid, "image_name": str(img_name), "question": question,
                "gt_answer": gt, "pred_answer": "", "em": 0,
                "is_yesno": is_yesno(gt), "error": f"image_open_failed:{e}"
            })
            continue

        inputs = processor(image, question, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成答案（BLIP VQA 常用 generate）
        # max_new_tokens 调小一点更稳
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            out_ids = model.generate(**inputs, max_new_tokens=10)

        pred = processor.decode(out_ids[0], skip_special_tokens=True)
        pred_n = norm_answer(pred)
        gt_n = norm_answer(gt)
        em = 1 if pred_n == gt_n else 0

        rows.append({
            "qid": qid,
            "image_name": str(img_name),
            "question": question,
            "gt_answer": gt,
            "pred_answer": pred,
            "em": em,
            "is_yesno": is_yesno(gt),
            "error": ""
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[OK] Saved:", out_csv)

    # ===== Metrics =====
    overall_em = df["em"].mean() if len(df) else 0.0
    df_yesno = df[df["is_yesno"] == True]
    df_open = df[df["is_yesno"] == False]

    yesno_em = df_yesno["em"].mean() if len(df_yesno) else 0.0
    open_em = df_open["em"].mean() if len(df_open) else 0.0

    # yes/no 额外算一个 accuracy（其实就是 EM，因为标签只有 yes/no）
    metrics_text = []
    metrics_text.append("Evaluation Metrics")
    metrics_text.append("=" * 28)
    metrics_text.append(f"Total samples: {len(df)}")
    metrics_text.append(f"Overall EM: {overall_em*100:.2f}%")
    metrics_text.append("")
    metrics_text.append(f"Yes/No samples: {len(df_yesno)} | Yes/No EM: {yesno_em*100:.2f}%")
    metrics_text.append(f"Open-ended samples: {len(df_open)} | Open EM: {open_em*100:.2f}%")

    out_metrics.write_text("\n".join(metrics_text), encoding="utf-8")
    print("[OK] Saved:", out_metrics)

    # ===== 5 cases =====
    # 取 3 条错的 + 2 条对的（如果数量不够就尽量凑）
    wrong = df[df["em"] == 0].head(3)
    right = df[df["em"] == 1].head(2)
    cases = pd.concat([wrong, right], ignore_index=True)

    lines = []
    lines.append("5 Example Cases (3 wrong + 2 correct)")
    lines.append("=" * 36)
    for idx, r in cases.iterrows():
        lines.append(f"[Case {idx+1}] qid={r['qid']} | image={r['image_name']} | yesno={r['is_yesno']}")
        lines.append(f"Q: {r['question']}")
        lines.append(f"GT: {r['gt_answer']}")
        lines.append(f"Pred: {r['pred_answer']}")
        lines.append(f"EM: {r['em']}")
        if r.get("error"):
            lines.append(f"Error: {r['error']}")
        lines.append("-" * 36)

    out_cases.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] Saved:", out_cases)

    print("\nDone.")
    print(f"Overall EM={overall_em:.4f} | YesNo EM={yesno_em:.4f} | Open EM={open_em:.4f}")


if __name__ == "__main__":
    main()
