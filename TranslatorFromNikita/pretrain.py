import os
import re
import lzma
import numpy as np
import torch
from collections import OrderedDict


PAD = "<PAD>"
PAD_ID = 0
UNK = "<UNK>"
UNK_ID = 1
EMPTY = "<EMPTY>"
EMPTY_ID = 2
ROOT = "<ROOT>"
ROOT_ID = 3
VOCAB_PREFIX = [PAD, UNK, EMPTY, ROOT]


class BaseVocab:
    def __init__(self, data=None, lang="", idx=0, cutoff=0, lower=False):
        self.data = data
        self.lang = lang
        self.idx = idx
        self.cutoff = cutoff
        self.lower = lower
        if data is not None:
            self.build_vocab()
        self.state_attrs = ["lang", "idx", "cutoff", "lower", "_unit2id", "_id2unit"]

    def build_vocab(self):
        raise NotImplementedError()

    def state_dict(self):
        state = OrderedDict()
        for attr in self.state_attrs:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        new = cls()
        for attr, value in state_dict.items():
            setattr(new, attr, value)
        return new

    def normalize_unit(self, unit):
        if self.lower:
            return unit.lower()
        return unit

    def unit2id(self, unit):
        unit = self.normalize_unit(unit)
        if unit in self._unit2id:
            return self._unit2id[unit]
        else:
            return self._unit2id[UNK]

    def id2unit(self, id):
        return self._id2unit[id]

    def map(self, units):
        return [self.unit2id(x) for x in units]

    def unmap(self, ids):
        return [self.id2unit(x) for x in ids]

    def __len__(self):
        return len(self._id2unit)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.unit2id(key)
        elif isinstance(key, int) or isinstance(key, list):
            return self.id2unit(key)
        else:
            raise TypeError("Vocab key must be one of str, list, or int")

    def __contains__(self, key):
        return key in self._unit2id

    @property
    def size(self):
        return len(self)

class PretrainedWordVocab(BaseVocab):
    def build_vocab(self):
        self._id2unit = VOCAB_PREFIX + self.data
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class Pretrain:
    def __init__(self, filename=None, vec_filename=None, max_vocab=-1, save_to_file=True):
        self.filename = filename
        self._vec_filename = vec_filename
        self._max_vocab = max_vocab
        self._save_to_file = save_to_file

    @property
    def vocab(self):
        if not hasattr(self, "_vocab"):
            self._vocab, self._emb = self.load()
        return self._vocab

    @property
    def emb(self):
        if not hasattr(self, "_emb"):
            self._vocab, self._emb = self.load()
        return self._emb

    def load(self):
        if self.filename is not None and os.path.exists(self.filename):
            try:
                data = torch.load(self.filename, lambda storage, loc: storage)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                return self.read_pretrain()
            return PretrainedWordVocab.load_state_dict(data["vocab"]), data["emb"]
        else:
            return self.read_pretrain()

    def read_pretrain(self):
        if self._vec_filename is None:
            raise Exception("Vector file is not provided.")
        try:
            words, emb, failed = self.read_from_file(self._vec_filename, open_func=lzma.open)
        except lzma.LZMAError:
            words, emb, failed = self.read_from_file(self._vec_filename, open_func=open)
        if failed > 0:
            emb = emb[:-failed]
        if len(emb) - len(VOCAB_PREFIX) != len(words):
            raise Exception("Loaded number of vectors does not match number of words.")
        if self._max_vocab > len(VOCAB_PREFIX) and self._max_vocab < len(words):
            words = words[:self._max_vocab - len(VOCAB_PREFIX)]
            emb = emb[:self._max_vocab]
        vocab = PretrainedWordVocab(words)
        if self._save_to_file:
            assert self.filename is not None, "Filename must be provided to save pretrained vector to file."
            data = {"vocab": vocab.state_dict(), "emb": emb}
            try:
                torch.save(data, self.filename)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                ...
        return vocab, emb

    def read_from_file(self, filename, open_func=open):
        tab_space_pattern = re.compile(r"[ \t]+")
        first = True
        words = []
        failed = 0
        with open_func(filename, "rb") as f:
            for i, line in enumerate(f):
                try:
                    line = line.decode()
                except UnicodeDecodeError:
                    failed += 1
                    continue
                if first:
                    first = False
                    line = line.strip().split(" ")
                    rows, cols = [int(x) for x in line]
                    emb = np.zeros((rows + len(VOCAB_PREFIX), cols), dtype=np.float32)
                    continue
                line = tab_space_pattern.split((line.rstrip()))
                emb[i+len(VOCAB_PREFIX)-1-failed, :] = [float(x) for x in line[-cols:]]
                words.append(" ".join(line[:-cols]))
        return words, emb, failed