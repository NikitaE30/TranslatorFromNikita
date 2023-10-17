from TranslatorFromNikita import doc, utils
from TranslatorFromNikita._constants import *
from TranslatorFromNikita.processor import UDProcessor, register_processor
import random
import torch
from TranslatorFromNikita.data import get_long_tensor, sort_all
from TranslatorFromNikita.vocab import CharVocab, MultiVocab
from TranslatorFromNikita.vocab_models import TagVocab, MultiVocab
from TranslatorFromNikita.doc import *
from TranslatorFromNikita.trainer_models import Trainer as BaseTrainer
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from TranslatorFromNikita.packed_lstm import PackedLSTM
from TranslatorFromNikita.dropout import WordDropout, LockedDropout
from TranslatorFromNikita.char_model import CharacterModel, CharacterLanguageModel
import math
from numbers import Number
import numpy as np


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

class CRFLoss(torch.nn.Module):
    def __init__(self, num_tag, batch_average=True):
        super().__init__()
        self._transitions = torch.nn.Parameter(torch.zeros(num_tag, num_tag))
        self._batch_average = batch_average

    def forward(self, inputs, masks, tag_indices):
        self.bs, self.sl, self.nc = inputs.size()
        unary_scores = self.crf_unary_score(inputs, masks, tag_indices)
        binary_scores = self.crf_binary_score(inputs, masks, tag_indices)
        log_norm = self.crf_log_norm(inputs, masks, tag_indices)
        log_likelihood = unary_scores + binary_scores - log_norm
        loss = torch.sum(-log_likelihood)
        if self._batch_average:
            loss = loss / self.bs
        else:
            total = masks.eq(0).sum()
            loss = loss / (total + 1e-8)
        return loss, self._transitions

    def crf_unary_score(self, inputs, masks, tag_indices):
        flat_inputs = inputs.view(self.bs, -1)
        flat_tag_indices = tag_indices + \
                set_cuda(torch.arange(self.sl).long().unsqueeze(0) * self.nc, tag_indices.is_cuda)
        unary_scores = torch.gather(flat_inputs, 1, flat_tag_indices).view(self.bs, -1)
        unary_scores.masked_fill_(masks, 0)
        return unary_scores.sum(dim=1)
    
    def crf_binary_score(self, inputs, masks, tag_indices):
        nt = tag_indices.size(-1) - 1
        start_indices = tag_indices[:, :nt]
        end_indices = tag_indices[:, 1:]
        flat_transition_indices = start_indices * self.nc + end_indices
        flat_transition_indices = flat_transition_indices.view(-1)
        flat_transition_matrix = self._transitions.view(-1)
        binary_scores = torch.gather(flat_transition_matrix, 0, flat_transition_indices).view(self.bs, -1)
        score_masks = masks[:, 1:]
        binary_scores.masked_fill_(score_masks, 0)
        return binary_scores.sum(dim=1)

    def crf_log_norm(self, inputs, masks, tag_indices):
        start_inputs = inputs[:,0,:]
        rest_inputs = inputs[:,1:,:]
        rest_masks = masks[:,1:]
        alphas = start_inputs
        trans = self._transitions.unsqueeze(0)
        for i in range(rest_inputs.size(1)):
            transition_scores = alphas.unsqueeze(2) + trans
            new_alphas = rest_inputs[:,i,:] + log_sum_exp(transition_scores, dim=1)
            m = rest_masks[:,i].unsqueeze(1).expand_as(new_alphas)
            new_alphas.masked_scatter_(m, alphas.masked_select(m))
            alphas = new_alphas
        log_norm = log_sum_exp(alphas, dim=1)
        return log_norm

class NERTagger(torch.nn.Module):
    def __init__(self, args, vocab, emb_matrix=None):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []
        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)
        input_size = 0
        if self.args["word_emb_dim"] > 0:
            self.word_emb = torch.nn.Embedding(len(self.vocab["word"]), self.args["word_emb_dim"], PAD_ID)
            if emb_matrix is not None:
                self.init_emb(emb_matrix)
            if not self.args.get("emb_finetune", True):
                self.word_emb.weight.detach_()
            input_size += self.args["word_emb_dim"]
        if self.args["char"] and self.args["char_emb_dim"] > 0:
            if self.args["charlm"]:
                add_unsaved_module("charmodel_forward", CharacterLanguageModel.load(args["charlm_forward_file"], finetune=False))
                add_unsaved_module("charmodel_backward", CharacterLanguageModel.load(args["charlm_backward_file"], finetune=False))
            else:
                self.charmodel = CharacterModel(args, vocab, bidirectional=True, attention=False)
            input_size += self.args["char_hidden_dim"] * 2
        if self.args.get("input_transform", False):
            self.input_transform = torch.nn.Linear(input_size, input_size)
        else:
            self.input_transform = None
        self.taggerlstm = PackedLSTM(input_size, self.args["hidden_dim"], self.args["num_layers"], batch_first=True, \
                bidirectional=True, dropout=0 if self.args["num_layers"] == 1 else self.args["dropout"])
        self.drop_replacement = None
        self.taggerlstm_h_init = torch.nn.Parameter(torch.zeros(2 * self.args["num_layers"], 1, self.args["hidden_dim"]), requires_grad=False)
        self.taggerlstm_c_init = torch.nn.Parameter(torch.zeros(2 * self.args["num_layers"], 1, self.args["hidden_dim"]), requires_grad=False)
        num_tag = len(self.vocab["tag"])
        self.tag_clf = torch.nn.Linear(self.args["hidden_dim"]*2, num_tag)
        self.tag_clf.bias.data.zero_()
        self.crit = CRFLoss(num_tag)
        self.drop = torch.nn.Dropout(args["dropout"])
        self.worddrop = WordDropout(args["word_dropout"])
        self.lockeddrop = LockedDropout(args["locked_dropout"])

    def init_emb(self, emb_matrix):
        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = torch.from_numpy(emb_matrix)
        vocab_size = len(self.vocab["word"])
        dim = self.args["word_emb_dim"]
        assert emb_matrix.size() == (vocab_size, dim), \
            "Input embedding matrix must match size: {} x {}".format(vocab_size, dim)
        self.word_emb.weight.data.copy_(emb_matrix)

    def forward(self, word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        inputs = []
        if self.args["word_emb_dim"] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]
        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]
        if self.args["char"] and self.args["char_emb_dim"] > 0:
            if self.args.get("charlm", None):
                char_reps_forward = self.charmodel_forward.get_representation(chars[0], charoffsets[0], charlens, char_orig_idx)
                char_reps_forward = PackedSequence(char_reps_forward.data, char_reps_forward.batch_sizes)
                char_reps_backward = self.charmodel_backward.get_representation(chars[1], charoffsets[1], charlens, char_orig_idx)
                char_reps_backward = PackedSequence(char_reps_backward.data, char_reps_backward.batch_sizes)
                inputs += [char_reps_forward, char_reps_backward]
            else:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(char_reps.data, char_reps.batch_sizes)
                inputs += [char_reps]
        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        if self.args["word_dropout"] > 0:
            lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = pad(lstm_inputs)
        lstm_inputs = self.lockeddrop(lstm_inputs)
        lstm_inputs = pack(lstm_inputs).data
        if self.input_transform:
            lstm_inputs = self.input_transform(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(\
                self.taggerlstm_h_init.expand(2 * self.args["num_layers"], word.size(0), self.args["hidden_dim"]).contiguous(), \
                self.taggerlstm_c_init.expand(2 * self.args["num_layers"], word.size(0), self.args["hidden_dim"]).contiguous()))
        lstm_outputs = lstm_outputs.data
        lstm_outputs = self.drop(lstm_outputs)
        lstm_outputs = pad(lstm_outputs)
        lstm_outputs = self.lockeddrop(lstm_outputs)
        lstm_outputs = pack(lstm_outputs).data
        logits = pad(self.tag_clf(lstm_outputs)).contiguous()
        loss, trans = self.crit(logits, word_mask, tags)
        return loss, logits, trans

class Trainer(BaseTrainer):
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            self.load(model_file, args)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            self.args = args
            self.vocab = vocab
            self.model = NERTagger(args, vocab, emb_matrix=pretrain.emb)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = utils.get_optimizer(self.args["optim"], self.parameters, self.args["lr"], momentum=self.args["momentum"])

    def update(self, batch, eval=False):
        inputs, _, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, chars, tags = inputs
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _, _ = self.model(word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx)
        loss_val = loss.data.item()
        if eval:
            return loss_val
        loss.backward()
        torch.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, chars, tags = inputs
        self.model.eval()
        _, logits, trans = self.model(word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx)
        trans = trans.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :sentlens[i]], trans)
            tags = self.vocab["tag"].unmap(tags)
            tag_seqs += [tags]
        if unsort:
            tag_seqs = utils.unsort(tag_seqs, orig_idx)
        return tag_seqs

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

    def load(self, filename, args=None):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            raise
        self.args = checkpoint["config"]
        if args: self.args.update(args)
        self.vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
        self.model = NERTagger(self.args, self.vocab)
        self.model.load_state_dict(checkpoint["model"], strict=False)

class DataLoader:
    def __init__(self, doc, batch_size, args, pretrain=None, vocab=None, evaluation=False, preprocess_tags=True):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc
        self.preprocess_tags = preprocess_tags
        data = self.load_doc(self.doc)
        self.tags = [[w[1] for w in sent] for sent in data]
        self.pretrain = pretrain
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab
        if args.get("sample_train", 1.0) < 1.0 and not self.eval:
            keep = int(args["sample_train"] * len(data))
            data = random.sample(data, keep)
        data = self.preprocess(data, self.vocab, args)
        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)
        self.data = self.chunk_batches(data)

    def init_vocab(self, data):
        def from_model(model_filename):
            state_dict = torch.load(model_filename, lambda storage, loc: storage)
            assert "vocab" in state_dict, "Cannot find vocab in charLM model file."
            return state_dict["vocab"]
        if self.eval:
            raise Exception("Vocab must exist for evaluation.")
        if self.args["charlm"]:
            charvocab = CharVocab.load_state_dict(from_model(self.args["charlm_forward_file"]))
        else: 
            charvocab = CharVocab(data, self.args["shorthand"])
        wordvocab = self.pretrain.vocab
        tagvocab = TagVocab(data, self.args["shorthand"], idx=1)
        vocab = MultiVocab({"char": charvocab,
                            "word": wordvocab,
                            "tag": tagvocab})
        return vocab

    def preprocess(self, data, vocab, args):
        processed = []
        if args.get("lowercase", True):
            case = lambda x: x.lower()
        else:
            case = lambda x: x
        if args.get("char_lowercase", False):
            char_case = lambda x: x.lower()
        else:
            char_case = lambda x: x
        for sent in data:
            processed_sent = [vocab["word"].map([case(w[0]) for w in sent])]
            processed_sent += [[vocab["char"].map([char_case(x) for x in w[0]]) for w in sent]]
            processed_sent += [vocab["tag"].map([w[1] for w in sent])]
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
        assert len(batch) == 3
        sentlens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, sentlens)
        sentlens = [len(x) for x in batch[0]]
        chars_forward, chars_backward, charoffsets_forward, charoffsets_backward, charlens = self.process_chars(batch[1])
        chars_sorted, char_orig_idx = sort_all([chars_forward, chars_backward, charoffsets_forward, charoffsets_backward], charlens)
        chars_forward, chars_backward, charoffsets_forward, charoffsets_backward = chars_sorted
        charlens = [len(sent) for sent in chars_forward]
        batch_words = [w for sent in batch[1] for w in sent]
        wordlens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], wordlens)
        batch_words = batch_words[0]
        wordlens = [len(x) for x in batch_words]
        words = get_long_tensor(batch[0], batch_size)
        words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(wordlens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)
        chars_forward = get_long_tensor(chars_forward, batch_size, pad_id=self.vocab["char"].unit2id(" "))
        chars_backward = get_long_tensor(chars_backward, batch_size, pad_id=self.vocab["char"].unit2id(" "))
        chars = torch.cat([chars_forward.unsqueeze(0), chars_backward.unsqueeze(0)])
        charoffsets = [charoffsets_forward, charoffsets_backward]
        tags = get_long_tensor(batch[2], batch_size)
        return words, words_mask, wordchars, wordchars_mask, chars, tags, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def load_doc(self, doc):
        data = doc.get([TEXT, NER], as_sentences=True, from_token=True)
        if self.preprocess_tags:
            data = self.process_tags(data)
        return data

    def process_tags(self, sentences):
        res = []
        convert_to_bioes = False
        is_bio = is_bio_scheme([x[1] for sent in sentences for x in sent])
        if is_bio and self.args.get("scheme", "bio").lower() == "bioes":
            convert_to_bioes = True
        for sent in sentences:
            words, tags = zip(*sent)
            if any([x is None or x == "_" for x in tags]):
                raise Exception("NER tag not found for some input data.")
            tags = to_bio2(tags)
            if convert_to_bioes:
                tags = bio2_to_bioes(tags)
            res.append([[w,t] for w,t in zip(words, tags)])
        return res

    def process_chars(self, sents):
        start_id, end_id = self.vocab["char"].unit2id("\n"), self.vocab["char"].unit2id(" ")
        chars_forward, chars_backward, charoffsets_forward, charoffsets_backward = [], [], [], []
        for sent in sents:
            chars_forward_sent, chars_backward_sent, charoffsets_forward_sent, charoffsets_backward_sent = [start_id], [start_id], [], []
            for word in sent:
                chars_forward_sent += word
                charoffsets_forward_sent = charoffsets_forward_sent + [len(chars_forward_sent)]
                chars_forward_sent += [end_id]
            for word in sent[::-1]:
                chars_backward_sent += word[::-1]
                charoffsets_backward_sent = [len(chars_backward_sent)] + charoffsets_backward_sent
                chars_backward_sent += [end_id]
            chars_forward.append(chars_forward_sent)
            chars_backward.append(chars_backward_sent)
            charoffsets_forward.append(charoffsets_forward_sent)
            charoffsets_backward.append(charoffsets_backward_sent)
        charlens = [len(sent) for sent in chars_forward]
        return chars_forward, chars_backward, charoffsets_forward, charoffsets_backward, charlens

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        random.shuffle(data)
        self.data = self.chunk_batches(data)

    def chunk_batches(self, data):
        data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        return data

def viterbi_decode(scores, transition_params):
    trellis = np.zeros_like(scores)
    backpointers = np.zeros_like(scores, dtype=np.int32)
    trellis[0] = scores[0]
    for t in range(1, scores.shape[0]):
        v = np.expand_dims(trellis[t-1], 1) + transition_params
        trellis[t] = scores[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)
    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score

def log_sum_exp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var

def is_bio_scheme(all_tags):
    for tag in all_tags:
        if tag == "O":
            continue
        elif len(tag) > 2 and tag[:2] in ("B-", "I-"):
            continue
        else:
            return False
    return True

def to_bio2(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag[0] == "I":
            if i == 0 or tags[i-1] == "O" or tags[i-1][1:] != tag[1:]:
                new_tags.append("B" + tag[1:])
            else:
                new_tags.append(tag)
        else:
            new_tags.append(tag)
    return new_tags

def bio2_to_bioes(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        else:
            if len(tag) < 2:
                raise Exception(f"Invalid BIO2 tag found: {tag}")
            else:
                if tag[:2] == "I-":
                    if i+1 < len(tags) and tags[i+1][:2] == "I-":
                        new_tags.append(tag)
                    else:
                        new_tags.append("E-" + tag[2:])
                elif tag[:2] == "B-":
                    if i+1 < len(tags) and tags[i+1][:2] == "I-":
                        new_tags.append(tag)
                    else:
                        new_tags.append("S-" + tag[2:])
                else:
                    raise Exception(f"Invalid IOB tag found: {tag}")
    return new_tags

def unpack_batch(batch, use_cuda):
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:6]]
    else:
        inputs = batch[:6]
    orig_idx = batch[6]
    word_orig_idx = batch[7]
    char_orig_idx = batch[8]
    sentlens = batch[9]
    wordlens = batch[10]
    charlens = batch[11]
    charoffsets = batch[12]
    return inputs, orig_idx, word_orig_idx, char_orig_idx, sentlens, wordlens, charlens, charoffsets


@register_processor(name=NER)
class NERProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([NER])
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        args = {"charlm_forward_file": config["forward_charlm_path"], "charlm_backward_file": config["backward_charlm_path"]}
        self._trainer = Trainer(args=args, model_file=config["model_path"], use_cuda=use_gpu)

    def process(self, document):
        batch = DataLoader(
            document, self.config["batch_size"], self.config, vocab=self.vocab, evaluation=True, preprocess_tags=False)
        preds = []
        for _, b in enumerate(batch):
            preds += self.trainer.predict(b)
        batch.doc.set([doc.NER], [y for x in preds for y in x], to_token=True)
        return batch.doc