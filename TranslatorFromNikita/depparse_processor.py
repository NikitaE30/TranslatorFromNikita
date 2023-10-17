import torch
import random
import numpy as np
from torch import nn
from copy import copy
from TranslatorFromNikita.doc import *
from TranslatorFromNikita import utils
from TranslatorFromNikita._constants import *
from TranslatorFromNikita.utils import unsort
import torch.nn.functional as F
from collections import OrderedDict
from TranslatorFromNikita.pretrain import Pretrain
from TranslatorFromNikita.hlstm import HighwayLSTM
from TranslatorFromNikita.dropout import WordDropout
from TranslatorFromNikita.char_model import CharacterModel
from TranslatorFromNikita.biaffine import DeepBiaffineScorer
from TranslatorFromNikita.data import get_long_tensor, sort_all
from TranslatorFromNikita.xpos_vocab_factory import xpos_vocab_factory
from TranslatorFromNikita.trainer_models import Trainer as BaseTrainer
from TranslatorFromNikita.processor import UDProcessor, register_processor
from TranslatorFromNikita.vocab import CharVocab, WordVocab, FeatureVocab, MultiVocab
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


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

class CompositeVocab(BaseVocab):
    def __init__(self, data=None, lang="", idx=0, sep="", keyed=False):
        self.sep = sep
        self.keyed = keyed
        super().__init__(data, lang, idx=idx)
        self.state_attrs += ["sep", "keyed"]

    def unit2parts(self, unit):
        if self.sep == "":
            parts = [x for x in unit]
        else:
            parts = unit.split(self.sep)
        if self.keyed:
            if len(parts) == 1 and parts[0] == "_":
                return dict()
            parts = [x.split("=") for x in parts]
            parts = dict(parts)
        elif unit == "_":
            parts = []
        return parts

    def unit2id(self, unit):
        parts = self.unit2parts(unit)
        if self.keyed:
            return [self._unit2id[k].get(parts[k], UNK_ID) if k in parts else EMPTY_ID for k in self._unit2id]
        else:
            return [self._unit2id[i].get(parts[i], UNK_ID) if i < len(parts) else EMPTY_ID for i in range(len(self._unit2id))]

    def id2unit(self, id):
        items = []
        for v, k in zip(id, self._id2unit.keys()):
            if v == EMPTY_ID: continue
            if self.keyed:
                items.append("{}={}".format(k, self._id2unit[k][v]))
            else:
                items.append(self._id2unit[k][v])
        res = self.sep.join(items)
        if res == "":
            res = "_"
        return res

    def build_vocab(self):
        allunits = [w[self.idx] for sent in self.data for w in sent]
        if self.keyed:
            self._id2unit = dict()
            for u in allunits:
                parts = self.unit2parts(u)
                for key in parts:
                    if key not in self._id2unit:
                        self._id2unit[key] = copy(VOCAB_PREFIX)
                    if parts[key] not in self._id2unit[key]:
                        self._id2unit[key].append(parts[key])
            if len(self._id2unit) == 0:
                self._id2unit["_"] = copy(VOCAB_PREFIX)
        else:
            self._id2unit = dict()
            allparts = [self.unit2parts(u) for u in allunits]
            for parts in allparts:
                for i, p in enumerate(parts):
                    if i not in self._id2unit:
                        self._id2unit[i] = copy(VOCAB_PREFIX)
                    if i < len(parts) and p not in self._id2unit[i]:
                        self._id2unit[i].append(p)
            if len(self._id2unit) == 0:
                self._id2unit[0] = copy(VOCAB_PREFIX)
        self._id2unit = OrderedDict([(k, self._id2unit[k]) for k in sorted(self._id2unit.keys())])
        self._unit2id = {k: {w:i for i, w in enumerate(self._id2unit[k])} for k in self._id2unit}

    def lens(self):
        return [len(self._unit2id[k]) for k in self._unit2id]

class Parser(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []
        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)
        input_size = 0
        if self.args["word_emb_dim"] > 0:
            self.word_emb = nn.Embedding(len(vocab["word"]), self.args["word_emb_dim"], padding_idx=0)
            self.lemma_emb = nn.Embedding(len(vocab["lemma"]), self.args["word_emb_dim"], padding_idx=0)
            input_size += self.args["word_emb_dim"] * 2
        if self.args["tag_emb_dim"] > 0:
            self.upos_emb = nn.Embedding(len(vocab["upos"]), self.args["tag_emb_dim"], padding_idx=0)
            if not isinstance(vocab["xpos"], CompositeVocab):
                self.xpos_emb = nn.Embedding(len(vocab["xpos"]), self.args["tag_emb_dim"], padding_idx=0)
            else:
                self.xpos_emb = nn.ModuleList()
                for l in vocab["xpos"].lens():
                    self.xpos_emb.append(nn.Embedding(l, self.args["tag_emb_dim"], padding_idx=0))
            self.ufeats_emb = nn.ModuleList()
            for l in vocab["feats"].lens():
                self.ufeats_emb.append(nn.Embedding(l, self.args["tag_emb_dim"], padding_idx=0))
            input_size += self.args["tag_emb_dim"] * 2
        if self.args["char"] and self.args["char_emb_dim"] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args["char_hidden_dim"], self.args["transformed_dim"], bias=False)
            input_size += self.args["transformed_dim"]
        if self.args["pretrain"]:
            add_unsaved_module("pretrained_emb", nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args["transformed_dim"], bias=False)
            input_size += self.args["transformed_dim"]
        self.parserlstm = HighwayLSTM(input_size, self.args["hidden_dim"], self.args["num_layers"], batch_first=True, bidirectional=True, dropout=self.args["dropout"], rec_dropout=self.args["rec_dropout"], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * self.args["num_layers"], 1, self.args["hidden_dim"]))
        self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * self.args["num_layers"], 1, self.args["hidden_dim"]))
        self.unlabeled = DeepBiaffineScorer(2 * self.args["hidden_dim"], 2 * self.args["hidden_dim"], self.args["deep_biaff_hidden_dim"], 1, pairwise=True, dropout=args["dropout"])
        self.deprel = DeepBiaffineScorer(2 * self.args["hidden_dim"], 2 * self.args["hidden_dim"], self.args["deep_biaff_hidden_dim"], len(vocab["deprel"]), pairwise=True, dropout=args["dropout"])
        if args["linearization"]:
            self.linearization = DeepBiaffineScorer(2 * self.args["hidden_dim"], 2 * self.args["hidden_dim"], self.args["deep_biaff_hidden_dim"], 1, pairwise=True, dropout=args["dropout"])
        if args["distance"]:
            self.distance = DeepBiaffineScorer(2 * self.args["hidden_dim"], 2 * self.args["hidden_dim"], self.args["deep_biaff_hidden_dim"], 1, pairwise=True, dropout=args["dropout"])
        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
        self.drop = nn.Dropout(args["dropout"])
        self.worddrop = WordDropout(args["word_dropout"])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        inputs = []
        if self.args["pretrain"]:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]
        if self.args["word_emb_dim"] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            lemma_emb = self.lemma_emb(lemma)
            lemma_emb = pack(lemma_emb)
            inputs += [word_emb, lemma_emb]
        if self.args["tag_emb_dim"] > 0:
            pos_emb = self.upos_emb(upos)
            if isinstance(self.vocab["xpos"], CompositeVocab):
                for i in range(len(self.vocab["xpos"])):
                    pos_emb += self.xpos_emb[i](xpos[:, :, i])
            else:
                pos_emb += self.xpos_emb(xpos)
            pos_emb = pack(pos_emb)
            feats_emb = 0
            for i in range(len(self.vocab["feats"])):
                feats_emb += self.ufeats_emb[i](ufeats[:, :, i])
            feats_emb = pack(feats_emb)
            inputs += [pos_emb, feats_emb]
        if self.args["char"] and self.args["char_emb_dim"] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]
        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        lstm_outputs, _ = self.parserlstm(lstm_inputs, sentlens, hx=(self.parserlstm_h_init.expand(2 * self.args["num_layers"], word.size(0), self.args["hidden_dim"]).contiguous(), self.parserlstm_c_init.expand(2 * self.args["num_layers"], word.size(0), self.args["hidden_dim"]).contiguous()))
        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)
        unlabeled_scores = self.unlabeled(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
        deprel_scores = self.deprel(self.drop(lstm_outputs), self.drop(lstm_outputs))
        if self.args["linearization"] or self.args["distance"]:
            head_offset = torch.arange(word.size(1), device=head.device).view(1, 1, -1).expand(word.size(0), -1, -1) - torch.arange(word.size(1), device=head.device).view(1, -1, 1).expand(word.size(0), -1, -1)
        if self.args["linearization"]:
            lin_scores = self.linearization(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            unlabeled_scores += F.logsigmoid(lin_scores * torch.sign(head_offset).float()).detach()
        if self.args["distance"]:
            dist_scores = self.distance(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            dist_pred = 1 + F.softplus(dist_scores)
            dist_target = torch.abs(head_offset)
            dist_kld = -torch.log((dist_target.float() - dist_pred)**2/2 + 1)
            unlabeled_scores += dist_kld.detach()
        diag = torch.eye(head.size(-1)+1, dtype=torch.bool, device=head.device).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float("inf"))
        preds = []
        if self.training:
            unlabeled_scores = unlabeled_scores[:, 1:, :]
            unlabeled_scores = unlabeled_scores.masked_fill(word_mask.unsqueeze(1), -float("inf"))
            unlabeled_target = head.masked_fill(word_mask[:, 1:], -1)
            loss = self.crit(unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)), unlabeled_target.view(-1))
            deprel_scores = deprel_scores[:, 1:]
            deprel_scores = torch.gather(deprel_scores, 2, head.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, len(self.vocab["deprel"]))).view(-1, len(self.vocab["deprel"]))
            deprel_target = deprel.masked_fill(word_mask[:, 1:], -1)
            loss += self.crit(deprel_scores.contiguous(), deprel_target.view(-1))
            if self.args["linearization"]:
                lin_scores = torch.gather(lin_scores[:, 1:], 2, head.unsqueeze(2)).view(-1)
                lin_scores = torch.cat([-lin_scores.unsqueeze(1)/2, lin_scores.unsqueeze(1)/2], 1)
                lin_target = torch.gather((head_offset[:, 1:] > 0).long(), 2, head.unsqueeze(2))
                loss += self.crit(lin_scores.contiguous(), lin_target.view(-1))
            if self.args["distance"]:
                dist_kld = torch.gather(dist_kld[:, 1:], 2, head.unsqueeze(2))
                loss -= dist_kld.sum()
            loss /= wordchars.size(0)
        else:
            loss = 0
            preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
            preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())
        return loss, preds

class Trainer(BaseTrainer):
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            self.load(model_file, pretrain)
        else:
            self.args = args
            self.vocab = vocab
            self.model = Parser(args, vocab, emb_matrix=pretrain.emb if pretrain is not None else None)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = utils.get_optimizer(self.args["optim"], self.parameters, self.args["lr"], betas=(0.9, self.args["beta2"]), eps=1e-6)

    def update(self, batch, eval=False):
        inputs, _, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs
        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)
        head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in zip(preds[0], sentlens)] # remove attachment for the root
        deprel_seqs = [self.vocab["deprel"].unmap([preds[1][i][j+1][h] for j, h in enumerate(hs)]) for i, hs in enumerate(head_seqs)]
        pred_tokens = [[[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(sentlens[i]-1)] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split(".")[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
                "model": model_state,
                "vocab": self.vocab.state_dict(),
                "config": self.args
                }
        try:
            torch.save(params, filename)
        except BaseException:
            ...

    def load(self, filename, pretrain):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            raise
        self.args = checkpoint["config"]
        self.vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
        emb_matrix = None
        if self.args["pretrain"] and pretrain is not None:
            emb_matrix = pretrain.emb
        self.model = Parser(self.args, self.vocab, emb_matrix=emb_matrix)
        self.model.load_state_dict(checkpoint["model"], strict=False)

class DataLoader:
    def __init__(self, doc, batch_size, args, pretrain, vocab=None, evaluation=False, sort_during_eval=False, max_sentence_size=None):
        self.batch_size = batch_size
        self.max_sentence_size=max_sentence_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.sort_during_eval = sort_during_eval
        self.doc = doc
        data = self.load_doc(doc)
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab
        self.pretrain_vocab = None
        if pretrain is not None and args["pretrain"]:
            self.pretrain_vocab = pretrain.vocab
        if args.get("sample_train", 1.0) < 1.0 and not self.eval:
            keep = int(args["sample_train"] * len(data))
            data = random.sample(data, keep)
        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)
        self.data = self.chunk_batches(data)

    def init_vocab(self, data):
        assert self.eval == False
        charvocab = CharVocab(data, self.args["shorthand"])
        wordvocab = WordVocab(data, self.args["shorthand"], cutoff=7, lower=True)
        uposvocab = WordVocab(data, self.args["shorthand"], idx=1)
        xposvocab = xpos_vocab_factory(data, self.args["shorthand"])
        featsvocab = FeatureVocab(data, self.args["shorthand"], idx=3)
        lemmavocab = WordVocab(data, self.args["shorthand"], cutoff=7, idx=4, lower=True)
        deprelvocab = WordVocab(data, self.args["shorthand"], idx=6)
        vocab = MultiVocab({"char": charvocab,
                            "word": wordvocab,
                            "upos": uposvocab,
                            "xpos": xposvocab,
                            "feats": featsvocab,
                            "lemma": lemmavocab,
                            "deprel": deprelvocab})
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab, args):
        processed = []
        xpos_replacement = [[ROOT_ID] * len(vocab["xpos"])] if isinstance(vocab["xpos"], CompositeVocab) else [ROOT_ID]
        feats_replacement = [[ROOT_ID] * len(vocab["feats"])]
        for sent in data:
            processed_sent = [[ROOT_ID] + vocab["word"].map([w[0] for w in sent])]
            processed_sent += [[[ROOT_ID]] + [vocab["char"].map([x for x in w[0]]) for w in sent]]
            processed_sent += [[ROOT_ID] + vocab["upos"].map([w[1] for w in sent])]
            processed_sent += [xpos_replacement + vocab["xpos"].map([w[2] for w in sent])]
            processed_sent += [feats_replacement + vocab["feats"].map([w[3] for w in sent])]
            if pretrain_vocab is not None:
                processed_sent += [[ROOT_ID] + pretrain_vocab.map([w[0].lower() for w in sent])]
            else:
                processed_sent += [[ROOT_ID] + [PAD_ID] * len(sent)]
            processed_sent += [[ROOT_ID] + vocab["lemma"].map([w[4] for w in sent])]
            processed_sent += [[to_int(w[5], ignore_error=self.eval) for w in sent]]
            processed_sent += [vocab["deprel"].map([w[6] for w in sent])]
            processed.append(processed_sent)
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
        assert len(batch) == 9
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]
        word_lens = [len(x) for x in batch_words]
        words = batch[0]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)
        upos = get_long_tensor(batch[2], batch_size)
        xpos = get_long_tensor(batch[3], batch_size)
        ufeats = get_long_tensor(batch[4], batch_size)
        pretrained = get_long_tensor(batch[5], batch_size)
        sentlens = [len(x) for x in batch[0]]
        lemma = get_long_tensor(batch[6], batch_size)
        head = get_long_tensor(batch[7], batch_size)
        deprel = get_long_tensor(batch[8], batch_size)
        return words, words_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, orig_idx, word_orig_idx, sentlens, word_lens

    def load_doc(self, doc):
        data = doc.get([TEXT, UPOS, XPOS, FEATS, LEMMA, HEAD, DEPREL], as_sentences=True)
        data = self.resolve_none(data)
        return data

    def resolve_none(self, data):
        for sent_idx in range(len(data)):
            for tok_idx in range(len(data[sent_idx])):
                for feat_idx in range(len(data[sent_idx][tok_idx])):
                    if data[sent_idx][tok_idx][feat_idx] is None:
                        data[sent_idx][tok_idx][feat_idx] = "_"
        return data

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        self.data = self.chunk_batches(data)
        random.shuffle(self.data)

    def chunk_batches(self, data):
        batches, data_orig_idx = data_to_batches(data=data, batch_size=self.batch_size,
                                                 eval_mode=self.eval, sort_during_eval=self.sort_during_eval,
                                                 max_sentence_size=self.max_sentence_size)
        self.data_orig_idx = data_orig_idx
        return batches

def tarjan(tree):
    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    _index = [0]
    cycles = []
    def strong_connect(i):
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True
        dependents = np.where(np.equal(tree, i))[0]
        for j in dependents:
            if indices[j] == -1:
                strong_connect(j)
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            elif onstack[j]:
                lowlinks[i] = min(lowlinks[i], indices[j])
        if lowlinks[i] == indices[i]:
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)
        return
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles

def chuliu_edmonds(scores):
    np.fill_diagonal(scores, -float("inf"))
    scores[0] = -float("inf")
    scores[0,0] = 0
    tree = np.argmax(scores, axis=1)
    cycles = tarjan(tree)
    if not cycles:
        return tree
    else:
        cycle = cycles.pop()
        cycle_locs = np.where(cycle)[0]
        cycle_subtree = tree[cycle]
        cycle_scores = scores[cycle, cycle_subtree]
        cycle_score = cycle_scores.sum()
        noncycle = np.logical_not(cycle)
        noncycle_locs = np.where(noncycle)[0]
        metanode_head_scores = scores[cycle][:,noncycle] - cycle_scores[:,None] + cycle_score
        metanode_dep_scores = scores[noncycle][:,cycle]
        metanode_heads = np.argmax(metanode_head_scores, axis=0)
        metanode_deps = np.argmax(metanode_dep_scores, axis=1)
        subscores = scores[noncycle][:,noncycle]
        subscores = np.pad(subscores, ( (0,1) , (0,1) ), "constant")
        subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
        subscores[:-1,-1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]
        contracted_tree = chuliu_edmonds(subscores)
        cycle_head = contracted_tree[-1]
        contracted_tree = contracted_tree[:-1]
        new_tree = -np.ones_like(tree)
        contracted_subtree = contracted_tree < len(contracted_tree)
        new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[contracted_tree[contracted_subtree]]
        contracted_subtree = np.logical_not(contracted_subtree)
        new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
        new_tree[cycle_locs] = tree[cycle_locs]
        cycle_root = metanode_heads[cycle_head]
        new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
        return new_tree

def chuliu_edmonds_one_root(scores):
    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0]+1
    if len(roots_to_try) == 1:
        return tree
    def set_root(scores, root):
        root_score = scores[root,0]
        scores = np.array(scores)
        scores[1:,0] = -float("inf")
        scores[root] = -float("inf")
        scores[root,0] = 0
        return scores, root_score
    best_score, best_tree = -np.inf, None
    for root in roots_to_try:
        _scores, root_score = set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = (tree_probs).sum()+(root_score) if (tree_probs > -np.inf).all() else -np.inf
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree
    try:
        assert best_tree is not None
    except:
        raise
    return best_tree

def data_to_batches(data, batch_size, eval_mode, sort_during_eval, max_sentence_size):
    res = []

    if not eval_mode:
        # sort sentences (roughly) by length for better memory utilization
        data = sorted(data, key = lambda x: len(x[0]), reverse=random.random() > .5)
        data_orig_idx = None
    elif sort_during_eval:
        (data, ), data_orig_idx = sort_all([data], [len(x[0]) for x in data])
    else:
        data_orig_idx = None

    current = []
    currentlen = 0
    for x in data:
        if max_sentence_size is not None and len(x[0]) > max_sentence_size:
            if currentlen > 0:
                res.append(current)
                current = []
                currentlen = 0
            res.append([x])
        else:
            if len(x[0]) + currentlen > batch_size and currentlen > 0:
                res.append(current)
                current = []
                currentlen = 0
            current.append(x)
            currentlen += len(x[0])

    if currentlen > 0:
        res.append(current)

    return res, data_orig_idx

def unpack_batch(batch, use_cuda):
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:11]]
    else:
        inputs = batch[:11]
    orig_idx = batch[11]
    word_orig_idx = batch[12]
    sentlens = batch[13]
    wordlens = batch[14]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens

def to_int(string, ignore_error=False):
    try:
        res = int(string)
    except ValueError as err:
        if ignore_error:
            return 0
        else:
            raise err
    return res

@register_processor(name=DEPPARSE)
class DepparseProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([DEPPARSE])
    REQUIRES_DEFAULT = set([TOKENIZE, POS, LEMMA])
    
    def __init__(self, config, pipeline, use_gpu):
        self._pretagged = None
        super().__init__(config, pipeline, use_gpu)
    def _set_up_requires(self):
        self._pretagged = self._config.get("pretagged")
        if self._pretagged:
            self._requires = set()
        else:
            self._requires = self.__class__.REQUIRES_DEFAULT

    def _set_up_model(self, config, use_gpu):
        self._pretrain = Pretrain(config["pretrain_path"]) if "pretrain_path" in config else None
        self._trainer = Trainer(pretrain=self.pretrain, model_file=config["model_path"], use_cuda=use_gpu)

    def process(self, document):
        batch = DataLoader(document, self.config["batch_size"], self.config, self.pretrain, vocab=self.vocab, evaluation=True,
                           sort_during_eval=self.config.get("sort_during_eval", True), max_sentence_size=self.config.get("max_sentence_size", None))
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        if batch.data_orig_idx is not None:
            preds = unsort(preds, batch.data_orig_idx)
        batch.doc.set([HEAD, DEPREL], [y for x in preds for y in x])
        for sentence in batch.doc.sentences:
            sentence.build_dependencies()
        return batch.doc