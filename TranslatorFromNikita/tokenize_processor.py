from TranslatorFromNikita.trainer import Trainer
from TranslatorFromNikita._constants import *
from TranslatorFromNikita.processor import UDProcessor, register_processor
from TranslatorFromNikita import doc
import re
from bisect import bisect_right
from copy import copy
import json
import numpy as np
import random
import torch
from TranslatorFromNikita.vocab import Vocab
from TranslatorFromNikita.doc import *
import io


NEWLINE_WHITESPACE_RE = re.compile(r"\n\s*\n")
NUMERIC_RE = re.compile(r"^([\d]+[,\.]*)+$")
WHITESPACE_RE = re.compile(r"\s")
SPACE_RE = re.compile(r"\s")
SPACE_SPLIT_RE = re.compile(r"( *[^ ]+)")
FIELD_NUM = 10
ID = "id"
TEXT = "text"
LEMMA = "lemma"
UPOS = "upos"
XPOS = "xpos"
FEATS = "feats"
HEAD = "head"
DEPREL = "deprel"
DEPS = "deps"
MISC = "misc"
FIELD_TO_IDX = {ID: 0, TEXT: 1, LEMMA: 2, UPOS: 3, XPOS: 4, FEATS: 5, HEAD: 6, DEPREL: 7, DEPS: 8, MISC: 9}


class CoNLL:
    
    @staticmethod
    def load_conll(f, ignore_gapping=True):
        doc, sent = [], []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if len(sent) > 0:
                    doc.append(sent)
                    sent = []
            else:
                if line.startswith("#"):
                    continue
                array = line.split("\t")
                if ignore_gapping and "." in array[0]:
                    continue
                assert len(array) == FIELD_NUM, \
                        f"Cannot parse CoNLL line: expecting {FIELD_NUM} fields, {len(array)} found."
                sent += [array]
        if len(sent) > 0:
            doc.append(sent)
        return doc

    @staticmethod
    def convert_conll(doc_conll):
        doc_dict = []
        for sent_conll in doc_conll:
            sent_dict = []
            for token_conll in sent_conll:
                token_dict = CoNLL.convert_conll_token(token_conll)
                sent_dict.append(token_dict)
            doc_dict.append(sent_dict)
        return doc_dict

    @staticmethod
    def convert_conll_token(token_conll):
        token_dict = {}
        for field in FIELD_TO_IDX:
            value = token_conll[FIELD_TO_IDX[field]]
            if value != "_":
                if field == HEAD:
                    token_dict[field] = int(value)
                elif field == ID:
                    token_dict[field] = tuple(int(x) for x in value.split("-"))
                else:
                    token_dict[field] = value
            if token_conll[FIELD_TO_IDX[TEXT]] == "_":
                token_dict[TEXT] = token_conll[FIELD_TO_IDX[TEXT]]
                token_dict[LEMMA] = token_conll[FIELD_TO_IDX[LEMMA]]
        return token_dict

    @staticmethod
    def conll2dict(input_file=None, input_str=None, ignore_gapping=True):
        assert any([input_file, input_str]) and not all([input_file, input_str]), "either input input file or input string"
        if input_str:
            infile = io.StringIO(input_str)
        else:
            infile = open(input_file)
        doc_conll = CoNLL.load_conll(infile, ignore_gapping)
        doc_dict = CoNLL.convert_conll(doc_conll)
        return doc_dict

    @staticmethod
    def convert_dict(doc_dict):
        doc_conll = []
        for sent_dict in doc_dict:
            sent_conll = []
            for token_dict in sent_dict:
                token_conll = CoNLL.convert_token_dict(token_dict)
                sent_conll.append(token_conll)
            doc_conll.append(sent_conll)
        return doc_conll

    @staticmethod
    def convert_token_dict(token_dict):
        token_conll = ["_" for i in range(FIELD_NUM)]
        for key in token_dict:
            if key == ID:
                token_conll[FIELD_TO_IDX[key]] = "-".join([str(x) for x in token_dict[key]]) if isinstance(token_dict[key], tuple) else str(token_dict[key])
            elif key in FIELD_TO_IDX:
                token_conll[FIELD_TO_IDX[key]] = str(token_dict[key])
        if "-" not in token_conll[FIELD_TO_IDX[ID]] and HEAD not in token_dict:
            token_conll[FIELD_TO_IDX[HEAD]] = str((token_dict[ID] if isinstance(token_dict[ID], int) else token_dict[ID][0]) - 1) # evaluation script requires head: int
        return token_conll

    @staticmethod
    def conll_as_string(doc):
        return_string = ""
        for sent in doc:
            for ln in sent:
                return_string += ("\t".join(ln)+"\n")
            return_string += "\n"
        return return_string

    @staticmethod
    def dict2conll(doc_dict, filename):
        doc_conll = CoNLL.convert_dict(doc_dict)
        conll_string = CoNLL.conll_as_string(doc_conll)
        with open(filename, "w") as outfile:
            outfile.write(conll_string)
        return

class DataLoader:
    def __init__(self, args, input_files={"json": None, "txt": None, "label": None}, input_text=None, input_data=None, vocab=None, evaluation=False):
        self.args = args
        self.eval = evaluation
        json_file = input_files["json"]
        txt_file = input_files["txt"]
        label_file = input_files["label"]
        if input_data is not None:
            self.data = input_data
        elif json_file is not None:
            with open(json_file) as f:
                self.data = json.load(f)
        else:
            assert txt_file is not None or input_text is not None
            if input_text is None:
                with open(txt_file) as f:
                    text = "".join(f.readlines()).rstrip()
            else:
                text = input_text
            if label_file is not None:
                with open(label_file) as f:
                    labels = "".join(f.readlines()).rstrip()
            else:
                labels = "\n\n".join(["0" * len(pt.rstrip()) for pt in NEWLINE_WHITESPACE_RE.split(text)])

            self.data = [[(WHITESPACE_RE.sub(" ", char), int(label))
                    for char, label in zip(pt.rstrip(), pc) if not (args.get("skip_newline", False) and char == "\n")]
                    for pt, pc in zip(NEWLINE_WHITESPACE_RE.split(text), NEWLINE_WHITESPACE_RE.split(labels)) if len(pt.rstrip()) > 0]
        self.data = [filter_consecutive_whitespaces(x) for x in self.data]
        self.vocab = vocab if vocab is not None else self.init_vocab()
        self.sentences = [self.para_to_sentences(para) for para in self.data]
        self.init_sent_ids()

    def init_vocab(self):
        vocab = Vocab(self.data, self.args["lang"])
        return vocab

    def init_sent_ids(self):
        self.sentence_ids = []
        self.cumlen = [0]
        for i, para in enumerate(self.sentences):
            for j in range(len(para)):
                self.sentence_ids += [(i, j)]
                self.cumlen += [self.cumlen[-1] + len(self.sentences[i][j])]

    def para_to_sentences(self, para):
        res = []
        funcs = []
        for feat_func in self.args["feat_funcs"]:
            if feat_func == "end_of_para" or feat_func == "start_of_para":
                continue
            if feat_func == "space_before":
                func = lambda x: 1 if x.startswith(" ") else 0
            elif feat_func == "capitalized":
                func = lambda x: 1 if x[0].isupper() else 0
            elif feat_func == "all_caps":
                func = lambda x: 1 if x.isupper() else 0
            elif feat_func == "numeric":
                func = lambda x: 1 if (NUMERIC_RE.match(x) is not None) else 0
            else:
                raise Exception("Feature function \"{}\" is undefined.".format(feat_func))
            funcs.append(func)
        composite_func = lambda x: [f(x) for f in funcs]
        def process_sentence(sent):
            return [(self.vocab.unit2id(y[0]), y[1], y[2], y[0]) for y in sent]
        use_end_of_para = "end_of_para" in self.args["feat_funcs"]
        use_start_of_para = "start_of_para" in self.args["feat_funcs"]
        current = []
        for i, (unit, label) in enumerate(para):
            label1 = label if not self.eval else 0
            feats = composite_func(unit)
            if use_end_of_para:
                f = 1 if i == len(para)-1 else 0
                feats.append(f)
            if use_start_of_para:
                f = 1 if i == 0 else 0
                feats.append(f)
            current += [(unit, label, feats)]
            if label1 == 2 or label1 == 4:
                if len(current) <= self.args["max_seqlen"]:
                    res.append(process_sentence(current))
                current = []
        if len(current) > 0:
            if self.eval or len(current) <= self.args["max_seqlen"]:
                res.append(process_sentence(current))
        return res

    def __len__(self):
        return len(self.sentence_ids)

    def shuffle(self):
        for para in self.sentences:
            random.shuffle(para)
        self.init_sent_ids()

    def next(self, eval_offsets=None, unit_dropout=0.0):
        null_feats = [0] * len(self.sentences[0][0][0][2])
        def strings_starting(id_pair, offset=0, pad_len=self.args["max_seqlen"]):
            pid, sid = id_pair
            res = copy(self.sentences[pid][sid][offset:])
            assert self.eval or len(res) <= self.args["max_seqlen"], "The maximum sequence length {} is less than that of the longest sentence length ({}) in the data, consider increasing it! {}".format(self.args["max_seqlen"], len(res), " ".join(["{}/{}".format(*x) for x in self.sentences[pid][sid]]))
            for sid1 in range(sid+1, len(self.sentences[pid])):
                res += self.sentences[pid][sid1]
                if not self.eval and len(res) >= self.args["max_seqlen"]:
                    res = res[:self.args["max_seqlen"]]
                    break
            if unit_dropout > 0 and not self.eval:
                unkid = self.vocab.unit2id("<UNK>")
                res = [(unkid, x[1], x[2], "<UNK>") if random.random() < unit_dropout else x for x in res]
            if pad_len > 0 and len(res) < pad_len:
                padid = self.vocab.unit2id("<PAD>")
                res += [(padid, -1, null_feats, "<PAD>")] * (pad_len - len(res))
            return res
        if eval_offsets is not None:
            pad_len = 0
            for eval_offset in eval_offsets:
                if eval_offset < self.cumlen[-1]:
                    pair_id = bisect_right(self.cumlen, eval_offset) - 1
                    pair = self.sentence_ids[pair_id]
                    pad_len = max(pad_len, len(strings_starting(pair, offset=eval_offset-self.cumlen[pair_id], pad_len=0)))
            res = []
            pad_len += 1
            for eval_offset in eval_offsets:
                if eval_offset >= self.cumlen[-1]:
                    padid = self.vocab.unit2id("<PAD>")
                    res += [[(padid, -1, null_feats, "<PAD>")] * pad_len]
                    continue
                pair_id = bisect_right(self.cumlen, eval_offset) - 1
                pair = self.sentence_ids[pair_id]
                res += [strings_starting(pair, offset=eval_offset-self.cumlen[pair_id], pad_len=pad_len)]
        else:
            id_pairs = random.sample(self.sentence_ids, min(len(self.sentence_ids), self.args["batch_size"]))
            res = [strings_starting(pair) for pair in id_pairs]
        units = [[y[0] for y in x] for x in res]
        labels = [[y[1] for y in x] for x in res]
        features = [[y[2] for y in x] for x in res]
        raw_units = [[y[3] for y in x] for x in res]
        convert = lambda t: (torch.from_numpy(np.array(t[0], dtype=t[1])))
        units, labels, features = list(map(convert, [(units, np.int64), (labels, np.int64), (features, np.float32)]))
        return units, labels, features, raw_units

def para_to_chunks(text, char_level_pred):
    chunks = []
    preds = []
    lastchunk = ""
    lastpred = ""
    for idx in range(len(text)):
        if re.match(r"^\w$", text[idx], flags=re.UNICODE):
            lastchunk += text[idx]
        else:
            if len(lastchunk) > 0 and not re.match(r"^\W+$", lastchunk, flags=re.UNICODE):
                chunks += [lastchunk]
                assert len(lastpred) > 0
                preds += [int(lastpred)]
                lastchunk = ""
            if not re.match(r"^\s$", text[idx], flags=re.UNICODE):
                chunks += [text[idx]]
                preds += [int(char_level_pred[idx])]
            else:
                lastchunk += text[idx]
        lastpred = char_level_pred[idx]
    if len(lastchunk) > 0:
        chunks += [lastchunk]
        preds += [int(lastpred)]
    return list(zip(chunks, preds))

def paras_to_chunks(text, char_level_pred):
    return [para_to_chunks(re.sub(r"\s", " ", pt.rstrip()), pc) for pt, pc in zip(text.split("\n\n"), char_level_pred.split("\n\n"))]

def process_sentence(sentence, mwt_dict=None):
    sent = []
    i = 0
    for tok, p, additional_info in sentence:
        expansion = None
        if (p == 3 or p == 4) and mwt_dict is not None:
            if tok in mwt_dict:
                expansion = mwt_dict[tok][0]
            elif tok.lower() in mwt_dict:
                expansion = mwt_dict[tok.lower()][0]
        if expansion is not None:
            infostr = None if len(additional_info) == 0 else "|".join([f"{k}={additional_info[k]}" for k in additional_info])
            sent.append({ID: (i+1, i+len(expansion)), TEXT: tok})
            if infostr is not None: sent[-1][MISC] = infostr
            for etok in expansion:
                sent.append({ID: (i+1, ), TEXT: etok})
                i += 1
        else:
            if len(tok) <= 0:
                continue
            if p == 3 or p == 4:
                additional_info["MWT"] = "Yes"
            infostr = None if len(additional_info) == 0 else "|".join([f"{k}={additional_info[k]}" for k in additional_info])
            sent.append({ID: (i+1, ), TEXT: tok})
            if infostr is not None: sent[-1][MISC] = infostr
            i += 1
    return sent

def output_predictions(output_file, trainer, data_generator, vocab, mwt_dict, max_seqlen=1000, orig_text=None, no_ssplit=False):
    paragraphs = []
    for i, p in enumerate(data_generator.sentences):
        start = 0 if i == 0 else paragraphs[-1][2]
        length = sum([len(x) for x in p])
        paragraphs += [(i, start, start+length, length+1)]
    paragraphs = list(sorted(paragraphs, key=lambda x: x[3], reverse=True))
    all_preds = [None] * len(paragraphs)
    all_raw = [None] * len(paragraphs)
    eval_limit = max(3000, max_seqlen)
    batch_size = trainer.args["batch_size"]
    batches = int((len(paragraphs) + batch_size - 1) / batch_size)
    t = 0
    for i in range(batches):
        batchparas = paragraphs[i * batch_size : (i + 1) * batch_size]
        offsets = [x[1] for x in batchparas]
        t += sum([x[3] for x in batchparas])
        batch = data_generator.next(eval_offsets=offsets)
        raw = batch[3]
        N = len(batch[3][0])
        if N <= eval_limit:
            pred = np.argmax(trainer.predict(batch), axis=2)
        else:
            idx = [0] * len(batchparas)
            Ns = [p[3] for p in batchparas]
            pred = [[] for _ in batchparas]
            while True:
                ens = [min(N - idx1, eval_limit) for idx1, N in zip(idx, Ns)]
                en = max(ens)
                batch1 = batch[0][:, :en], batch[1][:, :en], batch[2][:, :en], [x[:en] for x in batch[3]]
                pred1 = np.argmax(trainer.predict(batch1), axis=2)
                for j in range(len(batchparas)):
                    sentbreaks = np.where((pred1[j] == 2) + (pred1[j] == 4))[0]
                    if len(sentbreaks) <= 0 or idx[j] >= Ns[j] - eval_limit:
                        advance = ens[j]
                    else:
                        advance = np.max(sentbreaks) + 1
                    pred[j] += [pred1[j, :advance]]
                    idx[j] += advance
                if all([idx1 >= N for idx1, N in zip(idx, Ns)]):
                    break
                batch = data_generator.next(eval_offsets=[x+y for x, y in zip(idx, offsets)])
            pred = [np.concatenate(p, 0) for p in pred]
        for j, p in enumerate(batchparas):
            len1 = len([1 for x in raw[j] if x != "<PAD>"])
            if pred[j][len1-1] < 2:
                pred[j][len1-1] = 2
            elif pred[j][len1-1] > 2:
                pred[j][len1-1] = 4
            all_preds[p[0]] = pred[j][:len1]
            all_raw[p[0]] = raw[j]
    offset = 0
    oov_count = 0
    doc = []
    text = SPACE_RE.sub(" ", orig_text) if orig_text is not None else None
    char_offset = 0
    use_la_ittb_shorthand = trainer.args["shorthand"] == "la_ittb"
    for j in range(len(paragraphs)):
        raw = all_raw[j]
        pred = all_preds[j]
        current_tok = ""
        current_sent = []
        for t, p in zip(raw, pred):
            if t == "<PAD>":
                break
            if use_la_ittb_shorthand and t in (":", ";"):
                p = 2
            offset += 1
            if vocab.unit2id(t) == vocab.unit2id("<UNK>"):
                oov_count += 1
            current_tok += t
            if p >= 1:
                tok = vocab.normalize_token(current_tok)
                assert "\t" not in tok, tok
                if len(tok) <= 0:
                    current_tok = ""
                    continue
                if orig_text is not None:
                    st = -1
                    for part in SPACE_SPLIT_RE.split(current_tok):
                        if len(part) == 0: continue
                        st0 = text.index(part, char_offset) - char_offset
                        lstripped = part.lstrip()
                        if st < 0:
                            st = char_offset + st0 + (len(part) - len(lstripped))
                        char_offset += st0 + len(part)
                    additional_info = {START_CHAR: st, END_CHAR: char_offset}
                else:
                    additional_info = dict()
                current_sent.append((tok, p, additional_info))
                current_tok = ""
                if (p == 2 or p == 4) and not no_ssplit:
                    doc.append(process_sentence(current_sent, mwt_dict))
                    current_sent = []
        assert(len(current_tok) == 0)
        if len(current_sent):
            doc.append(process_sentence(current_sent, mwt_dict))
    if output_file: CoNLL.dict2conll(doc, output_file)
    return oov_count, offset, all_preds, doc

def filter_consecutive_whitespaces(para):
    filtered = []
    for i, (char, label) in enumerate(para):
        if i > 0:
            if char == " " and para[i-1][0] == " ":
                continue
        filtered.append((char, label))
    return filtered

@register_processor(name=TOKENIZE)
class TokenizeProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([TOKENIZE])
    REQUIRES_DEFAULT = set([])
    MAX_SEQ_LENGTH_DEFAULT = 1000

    def _set_up_model(self, config, use_gpu):
        if config.get("pretokenized"):
            self._trainer = None
        else:
            self._trainer = Trainer(model_file=config["model_path"], use_cuda=use_gpu)

    def process_pre_tokenized_text(self, input_src):
        document = []
        if isinstance(input_src, str):
            sentences = [sent.strip().split() for sent in input_src.strip().split("\n") if len(sent.strip()) > 0]
        elif isinstance(input_src, list):
            sentences = input_src
        idx = 0
        for sentence in sentences:
            sent = []
            for token_id, token in enumerate(sentence):
                sent.append({doc.ID: (token_id + 1, ), doc.TEXT: token, doc.MISC: f"start_char={idx}|end_char={idx + len(token)}"})
                idx += len(token) + 1
            document.append(sent)
        raw_text = " ".join([" ".join(sentence) for sentence in sentences])
        return raw_text, document

    def process(self, document):
        assert isinstance(document, str) or (self.config.get("pretokenized") or self.config.get("no_ssplit", False)), \
            "If neither \"pretokenized\" or \"no_ssplit\" option is enabled, the input to the TokenizerProcessor must be a string."
        if self.config.get("pretokenized"):
            raw_text, document = self.process_pre_tokenized_text(document)
        elif hasattr(self, "_variant"):
            return self._variant.process(document)
        else:
            raw_text = "\n\n".join(document) if isinstance(document, list) else document
            if self.config.get("lang") == "vi":
                text = "\n\n".join([x for x in raw_text.split("\n\n")]).rstrip()
                dummy_labels = "\n\n".join(["0" * len(x) for x in text.split("\n\n")])
                data = paras_to_chunks(text, dummy_labels)
                batches = DataLoader(self.config, input_data=data, vocab=self.vocab, evaluation=True)
            else:
                batches = DataLoader(self.config, input_text=raw_text, vocab=self.vocab, evaluation=True)
            _, _, _, document = output_predictions(None, self.trainer, batches, self.vocab, None,
                                   self.config.get("max_seqlen", TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT),
                                   orig_text=raw_text,
                                   no_ssplit=self.config.get("no_ssplit", False))
        return doc.Document(document, raw_text)