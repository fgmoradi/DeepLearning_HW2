#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention, Encoder, Decoder, and Model wrapper for HW02 (S2VT-style).
Rewritten to remove any import-time dependency on i2wData.pickle.
Pass vocab_size from main.py when constructing DecoderNet.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.special import expit


# -------------------------
# Attention
# -------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        """
        hidden_state: (1, B, H)
        encoder_outputs: (B, T, H)
        returns: context (B, H)
        """
        B, T, H = encoder_outputs.size()
        # expand hidden to (B, T, H)
        hidden_exp = hidden_state.view(B, 1, H).repeat(1, T, 1)
        # concat and score
        pair = torch.cat([encoder_outputs, hidden_exp], dim=2).view(-1, 2 * self.hidden_size)

        x = self.linear1(pair)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attn = self.to_weight(x).view(B, T)
        attn = F.softmax(attn, dim=1)

        ctx = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H)
        return ctx


# -------------------------
# Encoder (GRU over frame features)
# -------------------------
class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(512, 512, batch_first=True)

    def forward(self, x):
        """
        x: (B, T, 4096) video features
        returns:
          - output: (B, T, 512)
          - hidden: (1, B, 512)
        """
        B, T, D = x.size()
        x = x.view(-1, D)
        x = self.dropout(self.compress(x))
        x = x.view(B, T, 512)
        output, hidden = self.gru(x)
        return output, hidden


# -------------------------
# Decoder (GRU with Attention)
# -------------------------
class DecoderNet(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, vocab_size: int, word_dim: int, dropout_percentage: float = 0.3):
        """
        hidden_size: decoder GRU hidden size (match encoder, typically 512)
        output_size: size of final projection (usually == vocab_size)
        vocab_size: vocabulary size (including specials)
        word_dim: embedding dimension (e.g., 1024)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(self.vocab_size, self.word_dim)
        self.dropout = nn.Dropout(dropout_percentage)
        self.gru = nn.GRU(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def teacher_forcing_ratio(self, training_steps):
        # Keep same schedule as your original
        return expit(training_steps / 20 + 0.85)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        """
        Training forward:
          - targets: (B, L) token ids (with <SOS> at position 0)
        Inference forward (mode='inference'):
          - targets is ignored; greedy decode up to max length.
        Returns:
          seq_logProb: (B, L-1, V)
          seq_predictions: (B, L-1)
        """
        # Shapes
        _, B, _ = encoder_last_hidden_state.size()
        device = encoder_output.device

        # Hidden init
        dec_h = encoder_last_hidden_state  # (1, B, H)

        # Start token <SOS> == 1 (as per your preprocessing)
        y = Variable(torch.ones(B, 1, dtype=torch.long, device=device))

        seq_logProb = []

        if mode == 'train':
            # Embed all targets up-front
            # targets: (B, L) -> emb: (B, L, E)
            emb_targets = self.embedding(targets)
            L = emb_targets.size(1)
            for t in range(L - 1):
                thr = self.teacher_forcing_ratio(tr_steps)
                if random.uniform(0.05, 0.995) > thr:
                    # use ground-truth embedding at time t
                    inp = emb_targets[:, t]                    # (B, E)
                else:
                    # use previous prediction embedding
                    inp = self.embedding(y).squeeze(1)        # (B, E)

                ctx = self.attention(dec_h, encoder_output)    # (B, H)
                gru_in = torch.cat([inp, ctx], dim=1).unsqueeze(1)  # (B,1,E+H)
                out, dec_h = self.gru(gru_in, dec_h)                # out: (B,1,H)
                logits = self.to_final_output(out.squeeze(1))       # (B, V)
                seq_logProb.append(logits.unsqueeze(1))
                y = logits.argmax(dim=1, keepdim=True)              # next token id
        else:
            # Inference (greedy), fixed max length like original (28)
            max_len = 28
            for _ in range(max_len - 1):
                inp = self.embedding(y).squeeze(1)                 # (B, E)
                ctx = self.attention(dec_h, encoder_output)        # (B, H)
                gru_in = torch.cat([inp, ctx], dim=1).unsqueeze(1) # (B,1,E+H)
                out, dec_h = self.gru(gru_in, dec_h)
                logits = self.to_final_output(out.squeeze(1))      # (B, V)
                seq_logProb.append(logits.unsqueeze(1))
                y = logits.argmax(dim=1, keepdim=True)

        seq_logProb = torch.cat(seq_logProb, dim=1)    # (B, L-1, V)
        seq_predictions = seq_logProb.argmax(dim=2)    # (B, L-1)
        return seq_logProb, seq_predictions

    def infer(self, encoder_last_hidden_state, encoder_output):
        return self.forward(encoder_last_hidden_state, encoder_output, targets=None, mode='inference', tr_steps=None)


# -------------------------
# Wrapper
# -------------------------
class ModelMain(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        enc_out, enc_h = self.encoder(avi_feat)
        if mode == 'train':
            return self.decoder(encoder_last_hidden_state=enc_h,
                                encoder_output=enc_out,
                                targets=target_sentences,
                                mode=mode,
                                tr_steps=tr_steps)
        else:
            return self.decoder.infer(encoder_last_hidden_state=enc_h,
                                      encoder_output=enc_out)
