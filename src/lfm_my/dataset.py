"""Torch dataset: JSONL pairs -> per-pass padded tagger batches.

Sample layout: [bos] + pieces[:max_len-1]. Targets align position-for-position with the
input; padding positions are -100 (ignored by CrossEntropyLoss).
"""
from __future__ import annotations

from typing import List

import torch
from torch.utils.data import Dataset

from lfm_my.convert import convert_pair, pieces_of
from lfm_my.modeling_gectagger import KEEP_ID
from lfm_my.tags import append_id


class GecDataset(Dataset):
    """pairs: [{"noisy": str, "correct": str}] (whitespace-tokenized). Multi-pass conversions
    expand into multiple training samples (pass k trains on pass k-1's output)."""

    def __init__(self, pairs: List[dict], tokenizer, vocab_size: int, max_len: int = 128):
        self.max_len = max_len
        self.skipped = 0
        self.samples = []
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1

        def enc_word(w):
            return tokenizer.encode(w, add_special_tokens=False)

        for rec in pairs:
            conv = convert_pair(rec["noisy"], rec["correct"], enc_word, vocab_size)
            if conv is None:
                self.skipped += 1
                continue
            if not conv.passes:                                  # clean pair -> all-$KEEP pass
                # Encode per WORD, exactly as convert_pair does for edited pairs. Encoding the
                # whole sentence here instead would give clean examples a different piece
                # sequence than edited ones on any tokenizer where a word's encoding depends on
                # what precedes it (GPT-2-style space markers).
                pieces = pieces_of(rec["correct"].split(), enc_word)
                pieces = pieces[:max_len - 1]
                self.samples.append({
                    "input_ids": [bos] + pieces,
                    "label_targets": [KEEP_ID] * (len(pieces) + 1),
                    "detect_targets": [0] * (len(pieces) + 1),
                })
                continue
            for inp, p in zip(conv.inputs, conv.passes):
                pieces = inp[:max_len - 1]
                tags = p.tags[:max_len - 1]
                det = p.detect[:max_len - 1]
                bos_label = append_id(p.bos_append, vocab_size) if p.bos_append is not None else KEEP_ID
                bos_detect = 1 if p.bos_append is not None else 0
                self.samples.append({
                    "input_ids": [bos] + pieces,
                    "label_targets": [bos_label] + tags,
                    "detect_targets": [bos_detect] + det,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_fn(batch: List[dict], pad_id: int) -> dict:
    T = max(len(s["input_ids"]) for s in batch)
    input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), T), dtype=torch.long)
    label_targets = torch.full((len(batch), T), -100, dtype=torch.long)
    detect_targets = torch.full((len(batch), T), -100, dtype=torch.long)
    for b, s in enumerate(batch):
        n = len(s["input_ids"])
        input_ids[b, :n] = torch.tensor(s["input_ids"], dtype=torch.long)
        attention_mask[b, :n] = 1
        label_targets[b, :n] = torch.tensor(s["label_targets"], dtype=torch.long)
        detect_targets[b, :n] = torch.tensor(s["detect_targets"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "label_targets": label_targets, "detect_targets": detect_targets}
