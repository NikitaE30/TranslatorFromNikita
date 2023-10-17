from collections import Counter, OrderedDict
from TranslatorFromNikita.pretrain import PretrainedWordVocab
from TranslatorFromNikita.vocab import CharVocab

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

class TagVocab(BaseVocab):
    def build_vocab(self):
        counter = Counter([w[self.idx] for sent in self.data for w in sent])
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}

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
                "PretrainedWordVocab": PretrainedWordVocab,
                "TagVocab": TagVocab}
        new = cls()
        assert "_key2class" in state_dict, "Cannot find class name mapping in state dict!"
        key2class = state_dict.pop("_key2class")
        for k,v in state_dict.items():
            classname = key2class[k]
            new[k] = class_dict[classname].load_state_dict(v)
        return new