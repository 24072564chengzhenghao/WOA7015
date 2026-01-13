import json
from pathlib import Path
import random

def main():
    data_dir = Path("data/vqa-rad")
    src_json = data_dir / "VQA_RAD Dataset Public.json"
    img_dir = data_dir / "VQA_RAD Image Folder"

    assert src_json.exists(), f"Missing: {src_json}"
    assert img_dir.exists(), f"Missing: {img_dir}"

    items = json.load(open(src_json, "r", encoding="utf-8"))

    # Convert to a simpler format for our scripts
    simple = []
    for it in items:
        simple.append({
            "image": it["image_name"],
            "question": it["question"],
            "answer": it["answer"],
            "question_type": it.get("question_type", None),
            "answer_type": it.get("answer_type", None),
        })

    # Fixed split (70/10/20)
    seed = 42
    random.seed(seed)
    random.shuffle(simple)

    n = len(simple)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)

    train = simple[:n_train]
    val = simple[n_train:n_train+n_val]
    test = simple[n_train+n_val:]

    (data_dir / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_dir / "val.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_dir / "test.json").write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total: {n} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("Saved: train.json, val.json, test.json")

if __name__ == "__main__":
    main()
