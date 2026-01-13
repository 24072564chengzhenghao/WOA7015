# MedVQA (VQA-RAD) Starter (Local 8GB GPU)

This starter helps you run **3 experiments** on VQA-RAD:
1) Baseline (DenseNet121 + GRU, answer classification)  
2) BLIP-2 zero-shot  
3) BLIP-2 4-bit QLoRA (optional; best on Linux/WSL2)

## 0) Put the dataset
Unzip your dataset and place files like this:

```
data/vqa-rad/
  VQA_RAD Dataset Public.json
  VQA_RAD Image Folder/
    synpic....jpg
```

## 1) Create train/val/test json
Run:
```bash
python src/prepare_vqarad.py
```

This generates:
- data/vqa-rad/train.json
- data/vqa-rad/val.json
- data/vqa-rad/test.json

and uses `image_name/question/answer` fields from the public json.

## 2) Baseline (train + eval)
```bash
python src/baseline_train.py
python src/baseline_eval.py
```

Outputs:
- outputs/baseline/best.pt
- outputs/baseline/test_predictions.csv

## 3) BLIP-2 zero-shot (no training)
```bash
python src/blip2_zeroshot.py
```

Output:
- outputs/blip2_zeroshot/test_predictions.csv

## 4) BLIP-2 QLoRA (optional)
> bitsandbytes is most reliable on Linux/WSL2.
```bash
python src/blip2_qlora_train.py
python src/blip2_eval.py
```

Output:
- outputs/blip2_qlora/test_predictions.csv

## Notes
- Default split: 70/10/20 with seed=42.
- For 8GB GPU: baseline batch=16 (or 8), QLoRA batch=1~2.
