import re
from sklearn.metrics import f1_score

def normalize_text(s) -> str:
    # s may be int/float/None
    if s is None:
        s = ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s)     # collapse spaces
    return s

def exact_match(preds, gts):
    hit = 0
    for p, g in zip(preds, gts):
        if normalize_text(p) == normalize_text(g):
            hit += 1
    return hit / max(1, len(preds))

def accuracy(preds, gts):
    return exact_match(preds, gts)

def macro_f1_yesno(preds, gts):
    mp = [normalize_text(p) for p in preds]
    mg = [normalize_text(g) for g in gts]
    labels = ["yes", "no"]
    mp = [x if x in labels else "no" for x in mp]
    mg = [x if x in labels else "no" for x in mg]
    return f1_score(mg, mp, labels=labels, average="macro")
