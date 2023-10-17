from TranslatorFromNikita.pretrain import Pretrain
from TranslatorFromNikita._constants import *
from TranslatorFromNikita.processor import UDProcessor, register_processor
import torch
from TranslatorFromNikita.vocab import MultiVocab
from TranslatorFromNikita.doc import *
from copy import copy
from collections import OrderedDict
from TranslatorFromNikita.trainer_models import Trainer as BaseTrainer
from TranslatorFromNikita import utils
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from TranslatorFromNikita.biaffine import BiaffineScorer
from TranslatorFromNikita.hlstm import HighwayLSTM
from TranslatorFromNikita.dropout import WordDropout
from TranslatorFromNikita.char_model import CharacterModel


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

class Tagger(torch.nn.Module):
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
            self.word_emb = torch.nn.Embedding(len(vocab["word"]), self.args["word_emb_dim"], padding_idx=0)
            input_size += self.args["word_emb_dim"]
        if not share_hid:
            self.upos_emb = torch.nn.Embedding(len(vocab["upos"]), self.args["tag_emb_dim"], padding_idx=0)
        if self.args["char"] and self.args["char_emb_dim"] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = torch.nn.Linear(self.args["char_hidden_dim"], self.args["transformed_dim"], bias=False)
            input_size += self.args["transformed_dim"]
        if self.args["pretrain"]:    
            add_unsaved_module("pretrained_emb", torch.nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = torch.nn.Linear(emb_matrix.shape[1], self.args["transformed_dim"], bias=False)
            input_size += self.args["transformed_dim"]
        self.taggerlstm = HighwayLSTM(input_size, self.args["hidden_dim"], self.args["num_layers"], batch_first=True, bidirectional=True, dropout=self.args["dropout"], rec_dropout=self.args["rec_dropout"], highway_func=torch.tanh)
        self.drop_replacement = torch.nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = torch.nn.Parameter(torch.zeros(2 * self.args["num_layers"], 1, self.args["hidden_dim"]))
        self.taggerlstm_c_init = torch.nn.Parameter(torch.zeros(2 * self.args["num_layers"], 1, self.args["hidden_dim"]))
        self.upos_hid = torch.nn.Linear(self.args["hidden_dim"] * 2, self.args["deep_biaff_hidden_dim"])
        self.upos_clf = torch.nn.Linear(self.args["deep_biaff_hidden_dim"], len(vocab["upos"]))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()
        if share_hid:
            clf_constructor = lambda insize, outsize: torch.nn.Linear(insize, outsize)
        else:
            self.xpos_hid = torch.nn.Linear(self.args["hidden_dim"] * 2, self.args["deep_biaff_hidden_dim"] if not isinstance(vocab["xpos"], CompositeVocab) else self.args["composite_deep_biaff_hidden_dim"])
            self.ufeats_hid = torch.nn.Linear(self.args["hidden_dim"] * 2, self.args["composite_deep_biaff_hidden_dim"])
            clf_constructor = lambda insize, outsize: BiaffineScorer(insize, self.args["tag_emb_dim"], outsize)
        if isinstance(vocab["xpos"], CompositeVocab):
            self.xpos_clf = torch.nn.ModuleList()
            for l in vocab["xpos"].lens():
                self.xpos_clf.append(clf_constructor(self.args["composite_deep_biaff_hidden_dim"], l))
        else:
            self.xpos_clf = clf_constructor(self.args["deep_biaff_hidden_dim"], len(vocab["xpos"]))
            if share_hid:
                self.xpos_clf.weight.data.zero_()
                self.xpos_clf.bias.data.zero_()
        self.ufeats_clf = torch.nn.ModuleList()
        for l in vocab["feats"].lens():
            if share_hid:
                self.ufeats_clf.append(clf_constructor(self.args["deep_biaff_hidden_dim"], l))
                self.ufeats_clf[-1].weight.data.zero_()
                self.ufeats_clf[-1].bias.data.zero_()
            else:
                self.ufeats_clf.append(clf_constructor(self.args["composite_deep_biaff_hidden_dim"], l))
        self.crit = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.drop = torch.nn.Dropout(args["dropout"])
        self.worddrop = WordDropout(args["word_dropout"])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        inputs = []
        if self.args["word_emb_dim"] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]
        if self.args["pretrain"]:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]
        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]
        if self.args["char"] and self.args["char_emb_dim"] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]
        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(self.taggerlstm_h_init.expand(2 * self.args["num_layers"], word.size(0), self.args["hidden_dim"]).contiguous(), self.taggerlstm_c_init.expand(2 * self.args["num_layers"], word.size(0), self.args["hidden_dim"]).contiguous()))
        lstm_outputs = lstm_outputs.data
        upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        upos_pred = self.upos_clf(self.drop(upos_hid))
        preds = [pad(upos_pred).max(2)[1]]
        upos = pack(upos).data
        loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))
        if self.share_hid:
            xpos_hid = upos_hid
            ufeats_hid = upos_hid
            clffunc = lambda clf, hid: clf(self.drop(hid))
        else:
            xpos_hid = F.relu(self.xpos_hid(self.drop(lstm_outputs)))
            ufeats_hid = F.relu(self.ufeats_hid(self.drop(lstm_outputs)))
            if self.training:
                upos_emb = self.upos_emb(upos)
            else:
                upos_emb = self.upos_emb(upos_pred.max(1)[1])
            clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(upos_emb))
        xpos = pack(xpos).data
        if isinstance(self.vocab["xpos"], CompositeVocab):
            xpos_preds = []
            for i in range(len(self.vocab["xpos"])):
                xpos_pred = clffunc(self.xpos_clf[i], xpos_hid)
                loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos[:, i].view(-1))
                xpos_preds.append(pad(xpos_pred).max(2, keepdim=True)[1])
            preds.append(torch.cat(xpos_preds, 2))
        else:
            xpos_pred = clffunc(self.xpos_clf, xpos_hid)
            loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos.view(-1))
            preds.append(pad(xpos_pred).max(2)[1])
        ufeats_preds = []
        ufeats = pack(ufeats).data
        for i in range(len(self.vocab["feats"])):
            ufeats_pred = clffunc(self.ufeats_clf[i], ufeats_hid)
            loss += self.crit(ufeats_pred.view(-1, ufeats_pred.size(-1)), ufeats[:, i].view(-1))
            ufeats_preds.append(pad(ufeats_pred).max(2, keepdim=True)[1])
        preds.append(torch.cat(ufeats_preds, 2))
        return loss, preds

def unpack_batch(batch, use_cuda):
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:8]]
    else:
        inputs = batch[:8]
    orig_idx = batch[8]
    word_orig_idx = batch[9]
    sentlens = batch[10]
    wordlens = batch[11]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens

class Trainer(BaseTrainer):
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            self.load(model_file, pretrain)
        else:
            self.args = args
            self.vocab = vocab
            self.model = Tagger(args, vocab, emb_matrix=pretrain.emb if pretrain is not None else None, share_hid=args["share_hid"])
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = utils.get_optimizer(self.args["optim"], self.parameters, self.args["lr"], betas=(0.9, self.args["beta2"]), eps=1e-6)

    def update(self, batch, eval=False):
        inputs, _, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val
        loss.backward()
        torch.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained = inputs
        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens)
        upos_seqs = [self.vocab["upos"].unmap(sent) for sent in preds[0].tolist()]
        xpos_seqs = [self.vocab["xpos"].unmap(sent) for sent in preds[1].tolist()]
        feats_seqs = [self.vocab["feats"].unmap(sent) for sent in preds[2].tolist()]
        pred_tokens = [[[upos_seqs[i][j], xpos_seqs[i][j], feats_seqs[i][j]] for j in range(sentlens[i])] for i in range(batch_size)]
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
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            ...

    def load(self, filename, pretrain):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            raise
        self.args  = checkpoint["config"]
        self.vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
        emb_matrix = None
        if self.args["pretrain"] and pretrain is not None:
            emb_matrix = pretrain.emb
        self.model = Tagger(self.args, self.vocab, emb_matrix=emb_matrix, share_hid=self.args["share_hid"])
        self.model.load_state_dict(checkpoint["model"], strict=False)

@register_processor(name=POS)
class POSProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([POS])
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        self._pretrain = Pretrain(config["pretrain_path"]) if "pretrain_path" in config else None
        self._trainer  = Trainer(pretrain=self.pretrain, model_file=config["model_path"], use_cuda=use_gpu)