import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering


def norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def is_yesno(ans: str) -> bool:
    a = norm_text(ans)
    return a in ["yes", "no"]


def exact_match(p: str, g: str) -> int:
    return int(norm_text(p) == norm_text(g))


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    data_dir = Path("data/vqa-rad")
    image_dir = data_dir / "VQA_RAD Image Folder"
    test_json = data_dir / "test.json"

    out_dir = Path("outputs/vlm_blip_zeroshot_yesno")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "blip_yesno_predictions.csv"

    items = json.load(open(test_json, "r", encoding="utf-8"))
    if isinstance(items, dict) and "data" in items:
        items = items["data"]

    # filter yes/no only
    samples = []
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
        samples.append({
            "qid": str(ex.get("qid") or ex.get("question_id") or i),
            "image_name": str(img),
            "question": str(q),
            "gt_answer": str(a),
        })

    if len(samples) == 0:
        raise RuntimeError("No yes/no samples in test.json")

    model_name = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
    model.eval()

    rows = []
    hits = 0

    for s in tqdm(samples, desc="BLIP zero-shot (yes/no)"):
        img_path = image_dir / s["image_name"]
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, text=s["question"], return_tensors="pt").to(device)
        gen_ids = model.generate(**inputs, max_new_tokens=5)
        pred = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        em = exact_match(pred, s["gt_answer"])
        hits += em

        rows.append({
            "qid": s["qid"],
            "image_name": s["image_name"],
            "question": s["question"],
            "gt_answer": s["gt_answer"],
            "pred_answer": pred,
            "em": em
        })

    acc = hits / len(samples)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n=== BLIP Zero-shot YES/NO Test ===")
    print(f"Samples: {len(samples)}")
    print(f"Accuracy/EM: {acc:.4f}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
