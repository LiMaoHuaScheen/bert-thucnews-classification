#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chinese text classification: BERT-base-chinese fine-tune vs from-scratch vs TF-IDF baseline."""
import os, sys, json, time, argparse, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

CATEGORIES = ["education","entertainment","fashion","finance","game","politic","society","sport","stock","technology"]
BERT_LOCAL = "../hf_cache/bert-base-chinese"
OUT_DIR = "../output_text"
DATA_DIR = "../data"

def load_arrow(path):
    import pyarrow.ipc as ipc
    with ipc.open_stream(path) as f:
        t = f.read_all()
    texts = [t.column(1)[i].as_py() for i in range(t.num_rows)]
    labels = [int(t.column(0)[i].as_py()) for i in range(t.num_rows)]
    return texts, labels

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, self.labels[i]

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            b, y = batch
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**b)
            preds.extend(out.logits.argmax(-1).cpu().tolist())
            trues.extend(y.tolist())
    return accuracy_score(trues, preds)

def train_bert(init_mode, train_ds, val_ds, test_ds, device, epochs=3, lr=2e-5, batch=32, max_len=128):
    tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL)
    if init_mode == "scratch":
        config = BertForSequenceClassification.config_class.from_pretrained(BERT_LOCAL, num_labels=10)
        model = BertForSequenceClassification(config)
    else:
        model = BertForSequenceClassification.from_pretrained(BERT_LOCAL, num_labels=10)
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    hist = {"train_loss": [], "val_acc": [], "test_acc": []}
    for ep in range(epochs):
        model.train()
        tloss, nb = 0.0, 0
        for batch in train_loader:
            b, y = batch
            b = {k: v.to(device) for k, v in b.items()}
            b["labels"] = y.to(device)
            out = model(**b)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tloss += loss.item(); nb += 1
        va = evaluate(model, val_loader, device)
        te = evaluate(model, test_loader, device)
        hist["train_loss"].append(tloss/nb); hist["val_acc"].append(va); hist["test_acc"].append(te)
        print(f"  ep{ep+1}: loss={tloss/nb:.4f} val={va:.4f} test={te:.4f}", flush=True)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{init_mode}_epoch{ep+1}.pt"))
    return model, hist

def tfidf_svc(train_texts, train_labels, test_texts, test_labels):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    clf = LinearSVC(C=1.0, max_iter=2000)
    clf.fit(Xtr, train_labels)
    pred = clf.predict(Xte)
    return accuracy_score(test_labels, pred)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="limit train samples per class for quick test")
    ap.add_argument("--only", default="", help="only run one arm: ft|sc|tfidf|all")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Loading data...", flush=True)
    tr_texts, tr_labels = load_arrow(os.path.join(DATA_DIR, "thuc_news_text-train.arrow"))
    va_texts, va_labels = load_arrow(os.path.join(DATA_DIR, "thuc_news_text-validation.arrow"))
    te_texts, te_labels = load_arrow(os.path.join(DATA_DIR, "thuc_news_text-test.arrow"))
    print(f"  train {len(tr_texts)} val {len(va_texts)} test {len(te_texts)}", flush=True)
    if args.limit > 0:
        keep = []
        for c in range(10):
            ci = [i for i, l in enumerate(tr_labels) if l == c]
            keep.extend(ci[:args.limit])
        tr_texts = [tr_texts[i] for i in keep]; tr_labels = [tr_labels[i] for i in keep]
        print(f"  subsampled train {len(tr_texts)}", flush=True)
    if args.only in ("", "all", "tfidf"):
        print("TF-IDF + LinearSVC...", flush=True)
        tfidf_acc = tfidf_svc(tr_texts, tr_labels, te_texts, te_labels)
        print(f"  TF-IDF test acc: {tfidf_acc:.4f}", flush=True)
    else:
        tfidf_acc = None
    hist_ft = None; hist_sc = None
    if args.only in ("", "all", "ft", "sc"):
        tok = BertTokenizer.from_pretrained(BERT_LOCAL)
        tr_ds = TextDataset(tr_texts, tr_labels, tok, args.max_len)
        va_ds = TextDataset(va_texts, va_labels, tok, args.max_len)
        te_ds = TextDataset(te_texts, te_labels, tok, args.max_len)
        if args.only in ("", "all", "ft"):
            print("BERT fine-tune (pretrained)...", flush=True)
            _, hist_ft = train_bert("pretrained", tr_ds, va_ds, te_ds, device, args.epochs, 2e-5, args.batch, args.max_len)
        if args.only in ("", "all", "sc"):
            print("BERT from-scratch...", flush=True)
            _, hist_sc = train_bert("scratch", tr_ds, va_ds, te_ds, device, args.epochs, 2e-4, args.batch, args.max_len)
    summary = {"tfidf_acc": tfidf_acc, "finetune": hist_ft, "scratch": hist_sc}
    with open(os.path.join(OUT_DIR, "result_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()