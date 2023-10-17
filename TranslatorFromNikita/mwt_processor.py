from TranslatorFromNikita._constants import *
from TranslatorFromNikita.processor import UDProcessor, register_processor
import random
import torch
import TranslatorFromNikita.seq2seq_constant as constant
from TranslatorFromNikita.data import get_long_tensor, sort_all
from TranslatorFromNikita.vocab import Vocab


class DataLoader:
    def __init__(self, doc, batch_size, args, vocab=None, evaluation=False):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc
        data = self.load_doc(self.doc, evaluation=self.eval)
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab
        if args.get("sample_train", 1.0) < 1.0 and not self.eval:
            keep = int(args["sample_train"] * len(data))
            data = random.sample(data, keep)
        data = self.preprocess(data, self.vocab, args)
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def init_vocab(self, data):
        assert self.eval == False
        vocab = Vocab(data, self.args["shorthand"])
        return vocab

    def preprocess(self, data, vocab, args):
        processed = []
        for d in data:
            src = list(d[0])
            src = [constant.SOS] + src + [constant.EOS]
            src = vocab.map(src)
            if self.eval:
                tgt = src
            else:
                tgt = list(d[1])
            tgt_in = vocab.map([constant.SOS] + tgt)
            tgt_out = vocab.map(tgt + [constant.EOS])
            processed += [[src, tgt_in, tgt_out]]
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 3
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        assert tgt_in.size(1) == tgt_out.size(1), \
                "Target input and output sequence sizes do not match."
        return (src, src_mask, tgt_in, tgt_out, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_doc(self, doc, evaluation=False):
        data = doc.get_mwt_expansions(evaluation)
        if evaluation: data = [[e] for e in data]
        return data

@register_processor(MWT)
class MWTProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([MWT])
    REQUIRES_DEFAULT = set([TOKENIZE])

    def process(self, document):
        batch = DataLoader(document, self.config["batch_size"], self.config, vocab=self.vocab, evaluation=True)
        if len(batch) > 0:
            dict_preds = self.trainer.predict_dict(batch.doc.get_mwt_expansions(evaluation=True))
            if self.config["dict_only"]:
                preds = dict_preds
            else:
                preds = []
                for _, b in enumerate(batch):
                    preds += self.trainer.predict(b)
                if self.config.get("ensemble_dict", False):
                    preds = self.trainer.ensemble(batch.doc.get_mwt_expansions(evaluation=True), preds)
        else:
            preds = []
        batch.doc.set_mwt_expansions(preds)
        return batch.doc