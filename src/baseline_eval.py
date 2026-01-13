import torch
import pandas as pd
from pathlib import Path
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset_vqarad import VQARADDataset
from baseline_train import BaselineVQA
from eval_metrics import exact_match, accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    data_dir = Path("data/vqa-rad")
    image_dir = data_dir / "VQA_RAD Image Folder"
    test_path = data_dir / "test.json"

    ckpt = torch.load("outputs/baseline/best.pt", map_location="cpu")
    vocab = ckpt["vocab"]

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    ds = VQARADDataset(test_path, image_dir, transform=tfm, tokenizer=tokenizer)

    def collate(batch):
        imgs, toks, answers, metas = zip(*batch)
        imgs = torch.stack(imgs)
        input_ids = torch.stack([t["input_ids"] for t in toks])
        return imgs, input_ids, answers, metas

    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)

    model = BaselineVQA(num_answers=len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows, preds, gts = [], [], []
    with torch.no_grad():
        for imgs, input_ids, answers, metas in loader:
            imgs, input_ids = imgs.to(DEVICE), input_ids.to(DEVICE)
            logits = model(imgs, input_ids)
            p = logits.argmax(dim=1).cpu().tolist()
            p_text = [vocab[i] for i in p]

            for it, gt, pr in zip(metas, answers, p_text):
                rows.append({
                    "image": it["image"],
                    "question": it["question"],
                    "gt": gt,
                    "pred": pr
                })
            preds.extend(p_text)
            gts.extend(list(answers))

    out_dir = Path("outputs/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "test_predictions.csv", index=False)

    em = exact_match(preds, gts)
    acc = accuracy(preds, gts)
    print(f"Baseline Test: EM={em:.4f}, Acc={acc:.4f}")
    print("Saved:", out_dir / "test_predictions.csv")

if __name__ == "__main__":
    main()
