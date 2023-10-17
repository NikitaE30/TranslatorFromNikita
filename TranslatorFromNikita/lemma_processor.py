from TranslatorFromNikita import doc, utils
from TranslatorFromNikita._constants import *
from TranslatorFromNikita.processor import UDProcessor, register_processor
import random
import numpy as np
import torch
import TranslatorFromNikita.seq2seq_constant as constant
from TranslatorFromNikita.data import get_long_tensor, sort_all
from TranslatorFromNikita.doc import *
from collections import Counter, OrderedDict
from TranslatorFromNikita.seq2seq_model import Seq2SeqModel


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

class BaseMultiVocab:
    def __init__(self, vocab_dict=None):
        self._vocabs = OrderedDict()
        if vocab_dict is None:
            return
        assert all([isinstance(v, BaseVocab) for v in vocab_dict.values()])
        for k, v in vocab_dict.items():
            self._vocabs[k] = v

    def __setitem__(self, key, item):
        self._vocabs[key] = item

    def __getitem__(self, key):
        return self._vocabs[key]

    def state_dict(self):
        state = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        raise NotImplementedError

class Vocab(BaseVocab):
    def build_vocab(self):
        counter = Counter(self.data)
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class MultiVocab(BaseMultiVocab):
    @classmethod
    def load_state_dict(cls, state_dict):
        new = cls()
        for k,v in state_dict.items():
            new[k] = Vocab.load_state_dict(v)
        return new

def unpack_batch(batch, use_cuda):
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:6]]
    else:
        inputs = [b if b is not None else None for b in batch[:6]]
    orig_idx = batch[6]
    return inputs, orig_idx

class Trainer(object):
    def __init__(self, args=None, vocab=None, emb_matrix=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            self.load(model_file, use_cuda)
        else:
            self.args = args
            self.model = None if args["dict_only"] else Seq2SeqModel(args, emb_matrix=emb_matrix, use_cuda=use_cuda)
            self.vocab = vocab
            self.word_dict = dict()
            self.composite_dict = dict()

    def update(self, batch, eval=False):
        inputs, _ = unpack_batch(batch, self.use_cuda)
        src, src_mask, tgt_in, tgt_out, pos, edits = inputs
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs, edit_logits = self.model(src, src_mask, tgt_in, pos)
        if self.args.get("edit", False):
            assert edit_logits is not None
            loss = self.crit(log_probs.view(-1, self.vocab["char"].size), tgt_out.view(-1), \
                    edit_logits, edits)
        else:
            loss = self.crit(log_probs.view(-1, self.vocab["char"].size), tgt_out.view(-1))
        loss_val = loss.data.item()
        if eval:
            return loss_val
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, beam_size=1):
        inputs, orig_idx = unpack_batch(batch, self.use_cuda)
        src, src_mask, _, __, pos, edits = inputs
        self.model.eval()
        batch_size = src.size(0)
        preds, edit_logits = self.model.predict(src, src_mask, pos=pos, beam_size=beam_size)
        pred_seqs = [self.vocab["char"].unmap(ids) for ids in preds]
        pred_seqs = utils.prune_decoded_seqs(pred_seqs)
        pred_tokens = ["".join(seq) for seq in pred_seqs]
        pred_tokens = utils.unsort(pred_tokens, orig_idx)
        if self.args.get("edit", False):
            assert edit_logits is not None
            edits = np.argmax(edit_logits.data.cpu().numpy(), axis=1).reshape([batch_size]).tolist()
            edits = utils.unsort(edits, orig_idx)
        else:
            edits = None
        return pred_tokens, edits

    def update_lr(self, new_lr):
        utils.change_lr(self.optimizer, new_lr)

    def train_dict(self, triples):
        ctr = Counter()
        ctr.update([(p[0], p[1], p[2]) for p in triples])
        for p, _ in ctr.most_common():
            w, pos, l = p
            if (w,pos) not in self.composite_dict:
                self.composite_dict[(w,pos)] = l
            if w not in self.word_dict:
                self.word_dict[w] = l
        return

    def predict_dict(self, pairs):
        lemmas = []
        for p in pairs:
            w, pos = p
            if (w,pos) in self.composite_dict:
                lemmas += [self.composite_dict[(w,pos)]]
            elif w in self.word_dict:
                lemmas += [self.word_dict[w]]
            else:
                lemmas += [w]
        return lemmas

    def skip_seq2seq(self, pairs):
        skip = []
        for p in pairs:
            w, pos = p
            if (w,pos) in self.composite_dict:
                skip.append(True)
            elif w in self.word_dict:
                skip.append(True)
            else:
                skip.append(False)
        return skip

    def ensemble(self, pairs, other_preds):
        lemmas = []
        assert len(pairs) == len(other_preds)
        for p, pred in zip(pairs, other_preds):
            w, pos = p
            if (w,pos) in self.composite_dict:
                lemma = self.composite_dict[(w,pos)]
            elif w in self.word_dict:
                lemma = self.word_dict[w]
            else:
                lemma = pred
            if lemma is None:
                lemma = w
            lemmas.append(lemma)
        return lemmas

    def save(self, filename):
        params = {
                "model": self.model.state_dict() if self.model is not None else None,
                "dicts": (self.word_dict, self.composite_dict),
                "vocab": self.vocab.state_dict(),
                "config": self.args
                }
        try:
            torch.save(params, filename)
        except BaseException:
            ...

    def load(self, filename, use_cuda=False):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            raise
        self.args = checkpoint["config"]
        self.word_dict, self.composite_dict = checkpoint["dicts"]
        if not self.args["dict_only"]:
            self.model = Seq2SeqModel(self.args, use_cuda=use_cuda)
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model = None
        self.vocab = MultiVocab.load_state_dict(checkpoint["vocab"])

class DataLoader:
    def __init__(self, doc, batch_size, args, vocab=None, evaluation=False, conll_only=False, skip=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc
        data = self.load_doc(self.doc)
        if conll_only:
            return
        if skip is not None:
            assert len(data) == len(skip)
            data = [x for x, y in zip(data, skip) if not y]
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = dict()
            char_vocab, pos_vocab = self.init_vocab(data)
            self.vocab = MultiVocab({"char": char_vocab, "pos": pos_vocab})
        if args.get("sample_train", 1.0) < 1.0 and not self.eval:
            keep = int(args["sample_train"] * len(data))
            data = random.sample(data, keep)
        data = self.preprocess(data, self.vocab["char"], self.vocab["pos"], args)
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def init_vocab(self, data):
        assert self.eval is False, "Vocab file must exist for evaluation"
        char_data = "".join(d[0] + d[2] for d in data)
        char_vocab = Vocab(char_data, self.args["lang"])
        pos_data = [d[1] for d in data]
        pos_vocab = Vocab(pos_data, self.args["lang"])
        return char_vocab, pos_vocab

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
        assert len(batch) == 5
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        src = batch[0]
        src = get_long_tensor(src, batch_size)
        src_mask = torch.eq(src, constant.PAD_ID)
        tgt_in = get_long_tensor(batch[1], batch_size)
        tgt_out = get_long_tensor(batch[2], batch_size)
        pos = torch.LongTensor(batch[3])
        edits = torch.LongTensor(batch[4])
        assert tgt_in.size(1) == tgt_out.size(1), "Target input and output sequence sizes do not match."
        return src, src_mask, tgt_in, tgt_out, pos, edits, orig_idx

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_doc(self, doc):
        data = doc.get([TEXT, UPOS, LEMMA])
        data = self.resolve_none(data)
        return data

    def resolve_none(self, data):
        for tok_idx in range(len(data)):
            for feat_idx in range(len(data[tok_idx])):
                if data[tok_idx][feat_idx] is None:
                    data[tok_idx][feat_idx] = "_"
        return data

@register_processor(name=LEMMA)
class LemmaProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([LEMMA])
    REQUIRES_DEFAULT = set([TOKENIZE])
    DEFAULT_BATCH_SIZE = 5000
    
    def __init__(self, config, pipeline, use_gpu):
        self._use_identity = None
        super().__init__(config, pipeline, use_gpu)

    @property
    def use_identity(self):
        return self._use_identity

    def _set_up_model(self, config, use_gpu):
        if config.get("use_identity") in ["True", True]:
            self._use_identity = True
            self._config = config
            self.config["batch_size"] = LemmaProcessor.DEFAULT_BATCH_SIZE
        else:
            self._use_identity = False
            self._trainer = Trainer(model_file=config["model_path"], use_cuda=use_gpu)

    def _set_up_requires(self):
        if self.config.get("pos") and not self.use_identity:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT.union(set([POS]))
        else:
            self._requires = LemmaProcessor.REQUIRES_DEFAULT

    def process(self, document):
        if not self.use_identity:
            batch = DataLoader(document, self.config["batch_size"], self.config, vocab=self.vocab, evaluation=True)
        else:
            batch = DataLoader(document, self.config["batch_size"], self.config, evaluation=True, conll_only=True)
        if self.use_identity:
            preds = [word.text for sent in batch.doc.sentences for word in sent.words]
        elif self.config.get("dict_only", False):
            preds = self.trainer.predict_dict(batch.doc.get([doc.TEXT, doc.UPOS]))
        else:
            if self.config.get("ensemble_dict", False):
                skip = self.trainer.skip_seq2seq(batch.doc.get([doc.TEXT, doc.UPOS]))
                seq2seq_batch = DataLoader(document, self.config["batch_size"], self.config, vocab=self.vocab,
                                           evaluation=True, skip=skip)
            else:
                seq2seq_batch = batch
            preds = []
            edits = []
            for i, b in enumerate(seq2seq_batch):
                ps, es = self.trainer.predict(b, self.config["beam_size"])
                preds += ps
                if es is not None:
                    edits += es
            if self.config.get("ensemble_dict", False):
                preds = self.trainer.postprocess([x for x, y in zip(batch.doc.get([doc.TEXT]), skip) if not y], preds, edits=edits)
                i = 0
                preds1 = []
                for s in skip:
                    if s:
                        preds1.append("")
                    else:
                        preds1.append(preds[i])
                        i += 1
                preds = self.trainer.ensemble(batch.doc.get([doc.TEXT, doc.UPOS]), preds1)
            else:
                preds = self.trainer.postprocess(batch.doc.get([doc.TEXT]), preds, edits=edits)
        preds = [max([(len(x), x), (0, "_")])[1] for x in preds]
        batch.doc.set([doc.LEMMA], preds)
        return batch.doc