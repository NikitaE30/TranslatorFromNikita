import re
from copy import copy
from collections import Counter, OrderedDict
import TranslatorFromNikita.seq2seq_constant as constant


SPACE_RE = re.compile(r"\s")
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
        pairs = self.data
        allchars = "".join([src + tgt for src, tgt in pairs])
        counter = Counter(allchars)
        self._id2unit = constant.VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class CharVocab(BaseVocab):
    def build_vocab(self):
        if type(self.data[0][0]) is list:
            counter = Counter([c for sent in self.data for w in sent for c in w[self.idx]])
            for k in list(counter.keys()):
                if counter[k] < self.cutoff:
                    del counter[k]
        else:
            counter = Counter([c for sent in self.data for c in sent])
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: (counter[k], k), reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class WordVocab(BaseVocab):
    def __init__(self, data=None, lang="", idx=0, cutoff=0, lower=False, ignore=[]):
        self.ignore = ignore
        super().__init__(data, lang=lang, idx=idx, cutoff=cutoff, lower=lower)
        self.state_attrs += ["ignore"]

    def id2unit(self, id):
        if len(self.ignore) > 0 and id == EMPTY_ID:
            return "_"
        else:
            return super().id2unit(id)

    def unit2id(self, unit):
        if len(self.ignore) > 0 and unit in self.ignore:
            return self._unit2id[EMPTY]
        else:
            return super().unit2id(unit)

    def build_vocab(self):
        if self.lower:
            counter = Counter([w[self.idx].lower() for sent in self.data for w in sent])
        else:
            counter = Counter([w[self.idx] for sent in self.data for w in sent])
        for k in list(counter.keys()):
            if counter[k] < self.cutoff or k in self.ignore:
                del counter[k]
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

class XPOSVocab(CompositeVocab):
    def __init__(self, data=None, lang="", idx=0, sep="", keyed=False):
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

class FeatureVocab(CompositeVocab):
    def __init__(self, data=None, lang="", idx=0, sep="|", keyed=True):
        super().__init__(data, lang, idx=idx, sep=sep, keyed=keyed)

class MultiVocab(BaseMultiVocab):
    def state_dict(self):
        state = OrderedDict()
        key2class = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
            key2class[k] = type(v).__name__
        state["_key2class"] = key2class
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        class_dict = {"CharVocab": CharVocab,
                "WordVocab": WordVocab,
                "XPOSVocab": XPOSVocab,
                "FeatureVocab": FeatureVocab}
        new = cls()
        assert "_key2class" in state_dict, "Cannot find class name mapping in state dict!"
        key2class = state_dict.pop("_key2class")
        for k,v in state_dict.items():
            classname = key2class[k]
            new[k] = class_dict[classname].load_state_dict(v)
        return new

class Vocab(BaseVocab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang_replaces_spaces = any([self.lang.startswith(x) for x in ["zh", "ja", "ko"]])

    def build_vocab(self):
        paras = self.data
        counter = Counter()
        for para in paras:
            for unit in para:
                normalized = self.normalize_unit(unit[0])
                counter[normalized] += 1
        self._id2unit = [PAD, UNK] + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

    def normalize_unit(self, unit):
        normalized = unit
        if self.lang.startswith("vi"):
            normalized = normalized.lstrip()
        return normalized

    def normalize_token(self, token):
        token = SPACE_RE.sub(" ", token.lstrip())
        if self.lang_replaces_spaces:
            token = token.replace(" ", "")
        return token