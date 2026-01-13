import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class VQARADDataset(Dataset):
    def __init__(self, ann_path, image_dir, transform=None, tokenizer=None, max_len=32):
        self.ann_path = Path(ann_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(self.ann_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        # Each item should include:
        # { "image": "...jpg", "question": "...", "answer": "..." }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_path = self.image_dir / it["image"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        q = it["question"]
        a = it["answer"]

        if self.tokenizer is not None:
            tok = self.tokenizer(
                q,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            tok = {k: v.squeeze(0) for k, v in tok.items()}
            return image, tok, a, it

        return image, q, a, it
