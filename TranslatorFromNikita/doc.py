import io
import re
import json
import pickle


multi_word_token_id = re.compile(r"([0-9]+)-([0-9]+)")
multi_word_token_misc = re.compile(r".*MWT=Yes.*")
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
NER = "ner"
START_CHAR = "start_char"
END_CHAR = "end_char"
TYPE = "type"
SENTIMENT = "sentiment"


def decode_from_bioes(tags):
    res = []
    ent_idxs = []
    cur_type = None
    def flush():
        if len(ent_idxs) > 0:
            res.append({
                "start": ent_idxs[0], 
                "end": ent_idxs[-1], 
                "type": cur_type})
    for idx, tag in enumerate(tags):
        if tag is None:
            tag = "O"
        if tag == "O":
            flush()
            ent_idxs = []
        elif tag.startswith("B-"):
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
        elif tag.startswith("I-"):
            ent_idxs.append(idx)
            cur_type = tag[2:]
        elif tag.startswith("E-"):
            ent_idxs.append(idx)
            cur_type = tag[2:]
            flush()
            ent_idxs = []
        elif tag.startswith("S-"):
            flush()
            ent_idxs = [idx]
            cur_type = tag[2:]
            flush()
            ent_idxs = []
    flush()
    return res

def _readonly_setter(self, name):
    full_classname = self.__class__.__module__
    if full_classname is None:
        full_classname = self.__class__.__qualname__
    else:
        full_classname += "." + self.__class__.__qualname__
    raise ValueError(f"Property \"{name}\" of \"{full_classname}\" is read-only.")

class StanzaObject(object):

    @classmethod
    def add_property(cls, name, default=None, getter=None, setter=None):
        if hasattr(cls, name):
            raise ValueError(f"Property by the name of {name} already exists in {cls}. Maybe you want to find another name?")
        setattr(cls, f"_{name}", default)
        if getter is None:
            getter = lambda self: getattr(self, f"_{name}")
        if setter is None:
            setter = lambda self, value: _readonly_setter(self, name)
        setattr(cls, name, property(getter, setter))

class Document(StanzaObject):
    def __init__(self, sentences, text=None):
        self._sentences = []
        self._text = None
        self._num_tokens = 0
        self._num_words = 0
        self.text = text
        self._process_sentences(sentences)
        self._ents = []

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        self._sentences = value

    @property
    def num_tokens(self):
        return self._num_tokens

    @num_tokens.setter
    def num_tokens(self, value):
        self._num_tokens = value

    @property
    def num_words(self):
        return self._num_words

    @num_words.setter
    def num_words(self, value):
        self._num_words = value

    @property
    def ents(self):
        return self._ents

    @ents.setter
    def ents(self, value):
        self._ents = value

    @property
    def entities(self):
        return self._ents

    @entities.setter
    def entities(self, value):
        self._ents = value

    def _process_sentences(self, sentences):
        self.sentences = []
        for tokens in sentences:
            self.sentences.append(Sentence(tokens, doc=self))
            begin_idx, end_idx = self.sentences[-1].tokens[0].start_char, self.sentences[-1].tokens[-1].end_char
            if all([self.text is not None, begin_idx is not None, end_idx is not None]): self.sentences[-1].text = self.text[begin_idx: end_idx]
        self.num_tokens = sum([len(sentence.tokens) for sentence in self.sentences])
        self.num_words = sum([len(sentence.words) for sentence in self.sentences])

    def get(self, fields, as_sentences=False, from_token=False):
        if isinstance(fields, str):
            fields = [fields]
        assert isinstance(fields, list), "Must provide field names as a list."
        assert len(fields) >= 1, "Must have at least one field."
        results = []
        for sentence in self.sentences:
            cursent = []
            if from_token:
                units = sentence.tokens
            else:
                units = sentence.words
            for unit in units:
                if len(fields) == 1:
                    cursent += [getattr(unit, fields[0])]
                else:
                    cursent += [[getattr(unit, field) for field in fields]]
            if as_sentences:
                results.append(cursent)
            else:
                results += cursent
        return results

    def set(self, fields, contents, to_token=False, to_sentence=False):
        if isinstance(fields, str):
            fields = [fields]
        assert isinstance(fields, (tuple, list)), "Must provide field names as a list."
        assert isinstance(contents, (tuple, list)), "Must provide contents as a list (one item per line)."
        assert len(fields) >= 1, "Must have at least one field."
        assert not to_sentence or not to_token, "Both to_token and to_sentence set to True, which is very confusing"
        if to_sentence:
            assert len(self.sentences) == len(contents), \
                "Contents must have the same length as the sentences"
            for sentence, content in zip(self.sentences, contents):
                if len(fields) == 1:
                    setattr(sentence, fields[0], content)
                else:
                    for field, piece in zip(fields, content):
                        setattr(sentence, field, piece)
        else:
            assert (to_token and self.num_tokens == len(contents)) or self.num_words == len(contents), \
                "Contents must have the same length as the original file."
            cidx = 0
            for sentence in self.sentences:
                if to_token:
                    units = sentence.tokens
                else:
                    units = sentence.words
                for unit in units:
                    if len(fields) == 1:
                        setattr(unit, fields[0], contents[cidx])
                    else:
                        for field, content in zip(fields, contents[cidx]):
                            setattr(unit, field, content)
                    cidx += 1

    def set_mwt_expansions(self, expansions):
        idx_e = 0
        for sentence in self.sentences:
            idx_w = 0
            for token in sentence.tokens:
                idx_w += 1
                m = (len(token.id) > 1)
                n = multi_word_token_misc.match(token.misc) if token.misc is not None else None
                if not m and not n:
                    for word in token.words:
                        word.id = idx_w
                        word.head, word.deprel = None, None
                else:
                    expanded = [x for x in expansions[idx_e].split(" ") if len(x) > 0]
                    idx_e += 1
                    idx_w_end = idx_w + len(expanded) - 1
                    token.misc = None if token.misc == "MWT=Yes" else "|".join([x for x in token.misc.split("|") if x != "MWT=Yes"])
                    token.id = (idx_w, idx_w_end)
                    token.words = []
                    for i, e_word in enumerate(expanded):
                        token.words.append(Word({ID: idx_w + i, TEXT: e_word}))
                    idx_w = idx_w_end
            sentence._process_tokens(sentence.to_dict())
        self._process_sentences(self.to_dict())
        assert idx_e == len(expansions), "{} {}".format(idx_e, len(expansions))
        return

    def get_mwt_expansions(self, evaluation=False):
        expansions = []
        for sentence in self.sentences:
            for token in sentence.tokens:
                m = (len(token.id) > 1)
                n = multi_word_token_misc.match(token.misc) if token.misc is not None else None
                if m or n:
                    src = token.text
                    dst = " ".join([word.text for word in token.words])
                    expansions.append([src, dst])
        if evaluation: expansions = [e[0] for e in expansions]
        return expansions

    def build_ents(self):
        self.ents = []
        for s in self.sentences:
            s_ents = s.build_ents()
            self.ents += s_ents
        return self.ents

    def iter_words(self):
        for s in self.sentences:
            yield from s.words

    def iter_tokens(self):
        for s in self.sentences:
            yield from s.tokens

    def to_dict(self):
        return [sentence.to_dict() for sentence in self.sentences]

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_serialized(self):
        return pickle.dumps((self.text, self.to_dict()))

    @classmethod
    def from_serialized(cls, serialized_string):
        try:
            text, sentences = pickle.loads(serialized_string)
            doc = cls(sentences, text)
            doc.build_ents()
            return doc
        except:
            raise Exception(f"Could not create new Document from serialised string.")

class Sentence(StanzaObject):
    def __init__(self, tokens, doc=None):
        self._tokens = []
        self._words = []
        self._dependencies = []
        self._text = None
        self._ents = []
        self._doc = doc
        self._process_tokens(tokens)

    def _process_tokens(self, tokens):
        _, en = -1, -1
        self.tokens, self.words = [], []
        for i, entry in enumerate(tokens):
            if ID not in entry:
                entry[ID] = (i+1, )
            if isinstance(entry[ID], int):
                entry[ID] = (entry[ID], )
            m = (len(entry.get(ID)) > 1)
            n = multi_word_token_misc.match(entry.get(MISC)) if entry.get(MISC, None) is not None else None
            if m or n:
                if m: st, en = entry[ID]
                self.tokens.append(Token(entry))
            else:
                new_word = Word(entry)
                self.words.append(new_word)
                idx = entry.get(ID)[0]
                if idx <= en:
                    self.tokens[-1].words.append(new_word)
                else:
                    self.tokens.append(Token(entry, words=[new_word]))
                new_word.parent = self.tokens[-1]
        is_complete_dependencies = all(word.head is not None and word.deprel is not None for word in self.words)
        is_complete_words = (len(self.words) >= len(self.tokens)) and (len(self.words) == self.words[-1].id)
        if is_complete_dependencies and is_complete_words: self.build_dependencies()

    @property
    def doc(self):
        return self._doc

    @doc.setter
    def doc(self, value):
        self._doc = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def dependencies(self):
        return self._dependencies

    @dependencies.setter
    def dependencies(self, value):
        self._dependencies = value

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        self._tokens = value

    @property
    def words(self):
        return self._words

    @words.setter
    def words(self, value):
        self._words = value

    @property
    def ents(self):
        return self._ents

    @ents.setter
    def ents(self, value):
        self._ents = value

    @property
    def entities(self):
        return self._ents

    @entities.setter
    def entities(self, value):
        self._ents = value

    def build_ents(self):
        self.ents = []
        tags = [w.ner for w in self.tokens]
        decoded = decode_from_bioes(tags)
        for e in decoded:
            ent_tokens = self.tokens[e["start"]:e["end"]+1]
            self.ents.append(Span(tokens=ent_tokens, type=e["type"], doc=self.doc, sent=self))
        return self.ents

    @property
    def sentiment(self):
        return self._sentiment

    @sentiment.setter
    def sentiment(self, value):
        self._sentiment = value

    def build_dependencies(self):
        self.dependencies = []
        for word in self.words:
            if word.head == 0:
                word_entry = {ID: 0, TEXT: "ROOT"}
                head = Word(word_entry)
            else:
                head = self.words[word.head - 1]
                assert(word.head == head.id)
            self.dependencies.append((head, word.deprel, word))

    def print_dependencies(self, file=None):
        for dep_edge in self.dependencies:
            print((dep_edge[2].text, dep_edge[0].id, dep_edge[1]), file=file)

    def dependencies_string(self):
        dep_string = io.StringIO()
        self.print_dependencies(file=dep_string)
        return dep_string.getvalue().strip()

    def print_tokens(self, file=None):
        for tok in self.tokens:
            print(tok.pretty_print(), file=file)

    def tokens_string(self):
        toks_string = io.StringIO()
        self.print_tokens(file=toks_string)
        return toks_string.getvalue().strip()

    def print_words(self, file=None):
        for word in self.words:
            print(word.pretty_print(), file=file)

    def words_string(self):
        wrds_string = io.StringIO()
        self.print_words(file=wrds_string)
        return wrds_string.getvalue().strip()

    def to_dict(self):
        ret = []
        for token in self.tokens:
            ret += token.to_dict()
        return ret

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

class Token(StanzaObject):
    def __init__(self, token_entry, words=None):
        self._id = token_entry.get(ID)
        self._text = token_entry.get(TEXT)
        assert self._id and self._text, "id and text should be included for the token"
        self._misc = token_entry.get(MISC, None)
        self._ner = token_entry.get(NER, None)
        self._words = words if words is not None else []
        self._start_char = None
        self._end_char = None
        if self._misc is not None:
            self.init_from_misc()

    def init_from_misc(self):
        for item in self._misc.split("|"):
            key_value = item.split("=", 1)
            if len(key_value) == 1: continue
            key, value = key_value
            if key in (START_CHAR, END_CHAR):
                value = int(value)
            attr = f"_{key}"
            if hasattr(self, attr):
                setattr(self, attr, value)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def misc(self):
        return self._misc

    @misc.setter
    def misc(self, value):
        self._misc = value if self._is_null(value) == False else None

    @property
    def words(self):
        return self._words

    @words.setter
    def words(self, value):
        self._words = value
        for w in self._words:
            w.parent = self

    @property
    def start_char(self):
        return self._start_char

    @property
    def end_char(self):
        return self._end_char

    @property
    def ner(self):
        return self._ner

    @ner.setter
    def ner(self, value):
        self._ner = value if self._is_null(value) == False else None

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self, fields=[ID, TEXT, NER, MISC]):
        ret = []
        if len(self.id) > 1:
            token_dict = {}
            for field in fields:
                if getattr(self, field) is not None:
                    token_dict[field] = getattr(self, field)
            ret.append(token_dict)
        for word in self.words:
            word_dict = word.to_dict()
            if len(self.id) == 1 and NER in fields and getattr(self, NER) is not None: # propagate NER label to Word if it is a single-word token
                word_dict[NER] = getattr(self, NER)
            ret.append(word_dict)
        return ret

    def _is_null(self, value):
        return (value is None) or (value == "_")

class Word(StanzaObject):
    def __init__(self, word_entry):
        self._id = word_entry.get(ID, None)
        if isinstance(self._id, tuple):
            assert len(self._id) == 1
            self._id = self._id[0]
        self._text = word_entry.get(TEXT, None)
        assert self._id is not None and self._text is not None, "id and text should be included for the word. {}".format(word_entry)
        self._lemma = word_entry.get(LEMMA, None)
        self._upos = word_entry.get(UPOS, None)
        self._xpos = word_entry.get(XPOS, None)
        self._feats = word_entry.get(FEATS, None)
        self._head = word_entry.get(HEAD, None)
        self._deprel = word_entry.get(DEPREL, None)
        self._deps = word_entry.get(DEPS, None)
        self._misc = word_entry.get(MISC, None)
        self._parent = None
        if self._misc is not None:
            self.init_from_misc()

    def init_from_misc(self):
        for item in self._misc.split("|"):
            key_value = item.split("=", 1)
            if len(key_value) == 1: continue
            key, value = key_value
            attr = f"_{key}"
            if hasattr(self, attr):
                setattr(self, attr, value)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def lemma(self):
        return self._lemma

    @lemma.setter
    def lemma(self, value):
        self._lemma = value if self._is_null(value) == False or self._text == "_" else None

    @property
    def upos(self):
        return self._upos

    @upos.setter
    def upos(self, value):
        self._upos = value if self._is_null(value) == False else None

    @property
    def xpos(self):
        return self._xpos

    @xpos.setter
    def xpos(self, value):
        self._xpos = value if self._is_null(value) == False else None

    @property
    def feats(self):
        return self._feats

    @feats.setter
    def feats(self, value):
        self._feats = value if self._is_null(value) == False else None

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        self._head = int(value) if self._is_null(value) == False else None

    @property
    def deprel(self):
        return self._deprel

    @deprel.setter
    def deprel(self, value):
        self._deprel = value if self._is_null(value) == False else None

    @property
    def deps(self):
        return self._deps

    @deps.setter
    def deps(self, value):
        self._deps = value if self._is_null(value) == False else None

    @property
    def misc(self):
        return self._misc

    @misc.setter
    def misc(self, value):
        self._misc = value if self._is_null(value) == False else None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def pos(self):
        return self._upos

    @pos.setter
    def pos(self, value):
        self._upos = value if self._is_null(value) == False else None

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self, fields=[ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC]):
        word_dict = {}
        for field in fields:
            if getattr(self, field) is not None:
                word_dict[field] = getattr(self, field)
        return word_dict

    def pretty_print(self):
        features = [ID, TEXT, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL]
        feature_str = ";".join(["{}={}".format(k, getattr(self, k)) for k in features if getattr(self, k) is not None])
        return f"<{self.__class__.__name__} {feature_str}>"

    def _is_null(self, value):
        return (value is None) or (value == "_")

class Span(StanzaObject):
    def __init__(self, span_entry=None, tokens=None, type=None, doc=None, sent=None):
        assert span_entry is not None or (tokens is not None and type is not None), \
                "Either a span_entry or a token list needs to be provided to construct a span."
        assert doc is not None, "A parent doc must be provided to construct a span."
        self._text, self._type, self._start_char, self._end_char = [None] * 4
        self._tokens = []
        self._words = []
        self._doc = doc
        self._sent = sent
        if span_entry is not None:
            self.init_from_entry(span_entry)
        if tokens is not None:
            self.init_from_tokens(tokens, type)

    def init_from_entry(self, span_entry):
        self.text = span_entry.get(TEXT, None)
        self.type = span_entry.get(TYPE, None)
        self.start_char = span_entry.get(START_CHAR, None)
        self.end_char = span_entry.get(END_CHAR, None)

    def init_from_tokens(self, tokens, type):
        assert isinstance(tokens, list), "Tokens must be provided as a list to construct a span."
        assert len(tokens) > 0, "Tokens of a span cannot be an empty list."
        self.tokens = tokens
        self.type = type
        self.start_char = self.tokens[0].start_char
        self.end_char = self.tokens[-1].end_char
        self.text = self.doc.text[self.start_char:self.end_char]
        self.words = [w for t in tokens for w in t.words]

    @property
    def doc(self):
        return self._doc

    @doc.setter
    def doc(self, value):
        self._doc = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, value):
        self._tokens = value

    @property
    def words(self):
        return self._words

    @words.setter
    def words(self, value):
        self._words = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def start_char(self):
        return self._start_char

    @start_char.setter
    def start_char(self, value):
        self._start_char = value

    @property
    def end_char(self):
        return self._end_char

    @end_char.setter
    def end_char(self, value):
        self._end_char = value

    def to_dict(self):
        attrs = ["text", "type", "start_char", "end_char"]
        span_dict = dict([(attr_name, getattr(self, attr_name)) for attr_name in attrs])
        return span_dict

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def pretty_print(self):
        span_dict = self.to_dict()
        feature_str = ";".join(["{}={}".format(k,v) for k,v in span_dict.items()])
        return f"<{self.__class__.__name__} {feature_str}>"