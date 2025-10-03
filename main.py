#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, pickle, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import models  # EncoderNet, DecoderNet, ModelMain

# ------------------------------
# Reproducibility (safe defaults)
# ------------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

################################
# DATA PREPROCESSING
################################
class Dataprocessor(Dataset):
    """
    Loads (features, caption) training pairs.
    - label_file: path to features dir (e.g., 'training_data/feat')
    - files_dir: path to label json (e.g., 'training_label.json')
    """
    def __init__(self, label_file, files_dir, dictionary, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.avi = avi(label_file)               # dict: vid -> np.array(T,4096)
        self.w2i = w2i
        self.dictionary = dictionary
        self.data_pair = annotate(files_dir, dictionary, w2i)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        assert idx < len(self.data_pair)
        vid, sent_idx = self.data_pair[idx]
        data = torch.tensor(self.avi[vid], dtype=torch.float32)
        # keep your tiny noise augmentation (unchanged behavior)
        data = data + torch.rand_like(data) * 0.2
        return data, torch.tensor(sent_idx, dtype=torch.long)

class test_dataloader(Dataset):
    """Loads test .npy feature files and returns (id, features)."""
    def __init__(self, test_data_path):
        self.avi = []
        for file in os.listdir(test_data_path):
            if file.endswith(".npy"):
                key = file.split(".npy")[0]
                value = np.load(os.path.join(test_data_path, file))
                self.avi.append([key, value])

    def __len__(self):
        return len(self.avi)

    def __getitem__(self, idx):
        key, value = self.avi[idx]
        return key, torch.tensor(value, dtype=torch.float32)

def dictonaryFunc(word_min):
    # Load training captions
    with open('training_label.json', 'r') as f:
        file = json.load(f)

    word_count = {}
    for d in file:
        for s in d['caption']:
            word_sentence = re.sub(r'[.!,;?]', ' ', s).split()
            for word in word_sentence:
                word_count[word] = word_count.get(word, 0) + 1

    # Build dictionary with threshold
    dictionary = {w: c for w, c in word_count.items() if c > word_min}

    # Special tokens and mappings
    specials = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(specials): w for i, w in enumerate(dictionary)}
    w2i = {w: i + len(specials) for i, w in enumerate(dictionary)}
    for token, index in specials:
        i2w[index] = token
        w2i[token] = index
    return i2w, w2i, dictionary

def s_split(sentence, dictionary, w2i):
    words = re.sub(r'[.!,;?]', ' ', sentence).split()
    idxs = [w2i.get(w, 3) if w in dictionary else 3 for w in words]  # <UNK>=3
    return [1] + idxs + [2]  # <SOS>=1, <EOS>=2

def annotate(label_file, dictionary, w2i):
    with open(label_file, 'r') as f:
        label = json.load(f)
    annotated = []
    for d in label:
        for s in d['caption']:
            annotated.append((d['id'], s_split(s, dictionary, w2i)))
    return annotated

def word2index(w2i, w): return w2i[w]
def index2word(i2w, i): return i2w[i]
def sentence2index(w2i, sentence): return [w2i[w] for w in sentence]
def index2sentence(i2w, index_seq): return [i2w[int(i)] for i in index_seq]

def avi(files_dir):
    table = {}
    for file in os.listdir(files_dir):
        if file.endswith(".npy"):
            arr = np.load(os.path.join(files_dir, file))
            table[file.split('.npy')[0]] = arr
    return table

################################
# TRAIN / EVAL
################################
def calculate_loss(x, y, lengths, loss_fn):
    """
    x: (B, L-1, V) predicted logits
    y: (B, L-1) target indices (shifted to skip <SOS>)
    lengths: original lengths per sample
    """
    batch_size = x.size(0)
    predict_cat, ground_cat, first = None, None, True

    for b in range(batch_size):
        pred = x[b]
        gt = y[b]
        seq_len = lengths[b] - 1  # exclude <SOS>
        pred = pred[:seq_len]
        gt = gt[:seq_len]
        if first:
            predict_cat, ground_cat = pred, gt
            first = False
        else:
            predict_cat = torch.cat([predict_cat, pred], dim=0)
            ground_cat = torch.cat([ground_cat, gt], dim=0)

    loss = loss_fn(predict_cat, ground_cat)
    return loss  # keep same semantics as original

def train_one_epoch(model, epoch, train_loader, loss_func, device):
    model.train()
    print(f"Epoch {epoch}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for batch_idx, (avi_feats, ground_truths, lengths) in enumerate(train_loader):
        avi_feats, ground_truths = avi_feats.to(device), ground_truths.to(device)
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

        optimizer.zero_grad()
        seq_logProb, _ = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)

        # shift targets to drop <SOS>
        ground_truths = ground_truths[:, 1:]
        loss = calculate_loss(seq_logProb, ground_truths, lengths, loss_func)
        loss.backward()
        optimizer.step()

    print(f"Epoch:{epoch} & loss:{loss.item():.3f}")

def evaluate(val_loader, model, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (avi_feats, ground_truths, lengths) in enumerate(val_loader):
            avi_feats = avi_feats.to(device)
            _logp, _preds = model(avi_feats, mode='inference')
            break

################################
# MINIBATCH COLLATE
################################
def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    targets = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end].clone()
    return avi_data, targets, lengths

################################
# ARGPARSE + MAIN
################################
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_feat_dir', default='training_data/feat')
    ap.add_argument('--train_label_json', default='training_label.json')
    ap.add_argument('--test_feat_dir', default='testing_data/feat')
    ap.add_argument('--test_label_json', default='testing_label.json')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=5)
    ap.add_argument('--num_workers', type=int, default=8)  # set 0 if workers hang
    ap.add_argument('--save_dir', default='SavedModel')
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build vocab from training labels (threshold=4 as in your code)
    i2w, w2i, dictionary = dictonaryFunc(4)

    # Ensure save dir; persist i2w for inference time
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'i2wData.pickle'), 'wb') as f:
        pickle.dump(i2w, f)
    # (optional compatibility: also drop a copy at root)
    with open('i2wData.pickle', 'wb') as f:
        pickle.dump(i2w, f)

    # Datasets & loaders
    train_dataset = Dataprocessor(args.train_feat_dir, args.train_label_json, dictionary, w2i)
    val_dataset   = Dataprocessor(args.test_feat_dir,  args.test_label_json,  dictionary, w2i)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=minibatch)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False,
                              num_workers=args.num_workers, collate_fn=minibatch)

    # Model: pass vocab size into decoder (models.py should NOT read pickle at import)
    vocab_size = len(i2w) + 4
    encoder = models.EncoderNet()
    decoder = models.DecoderNet(hidden_size=512, output_size=vocab_size,
                                vocab_size=vocab_size, word_dim=1024, dropout_percentage=0.3)
    model = models.ModelMain(encoder=encoder, decoder=decoder).to(device)

    loss_fn = nn.CrossEntropyLoss()

    # Train
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, epoch, train_loader, loss_fn, device)
        evaluate(val_loader, model, device)
    end = time.time()

    # Save model
    torch.save(model, os.path.join(args.save_dir, 'model1.h5'))
    print(f"Training finished. Elapsed time: {end - start:.3f} seconds.")

if __name__ == "__main__":
    main()
