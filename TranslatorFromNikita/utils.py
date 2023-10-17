from __future__ import division, print_function
import unicodedata
import sys


lcode2lang = {
    "af": "Afrikaans",
    "grc": "Ancient_Greek",
    "ar": "Arabic",
    "hy": "Armenian",
    "eu": "Basque",
    "be": "Belarusian",
    "br": "Breton",
    "bg": "Bulgarian",
    "bxr": "Buryat",
    "ca": "Catalan",
    "zh-hant": "Traditional_Chinese",
    "lzh": "Classical_Chinese",
    "cop": "Coptic",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fo": "Faroese",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "got": "Gothic",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "kk": "Kazakh",
    "ko": "Korean",
    "kmr": "Kurmanji",
    "lt": "Lithuanian",
    "olo": "Livvi",
    "la": "Latin",
    "lv": "Latvian",
    "mt": "Maltese",
    "mr": "Marathi",
    "pcm": "Naija",
    "sme": "North_Sami",
    "nb": "Norwegian_Bokmaal",
    "nn": "Norwegian_Nynorsk",
    "cu": "Old_Church_Slavonic",
    "fro": "Old_French",
    "orv": "Old_Russian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "gd": "Scottish_Gaelic",
    "sr": "Serbian",
    "zh-hans": "Simplified_Chinese",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "swl": "Swedish_Sign_Language",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "hsb": "Upper_Sorbian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "vi": "Vietnamese",
    "wo": "Wolof"
}
lang2lcode = {lcode2lang[k]: k for k in lcode2lang}
langlower2lcode = {lcode2lang[k].lower(): k.lower() for k in lcode2lang}
lcode2lang["nb"] = "Norwegian"
lcode2lang["zh"] = "Simplified_Chinese"
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)
CONTENT_DEPRELS = {
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative",
    "expl", "dislocated", "advcl", "advmod", "discourse", "nmod", "appos",
    "nummod", "acl", "amod", "conj", "fixed", "flat", "compound", "list",
    "parataxis", "orphan", "goeswith", "reparandum", "root", "dep"
}
FUNCTIONAL_DEPRELS = {
    "aux", "cop", "mark", "det", "clf", "case", "cc"
}
UNIVERSAL_FEATURES = {
    "PronType", "NumType", "Poss", "Reflex", "Foreign", "Abbr", "Gender",
    "Animacy", "Number", "Case", "Definite", "Degree", "VerbForm", "Mood",
    "Tense", "Aspect", "Voice", "Evident", "Polarity", "Person", "Polite"
}


class UDError(Exception):
    ...
    
def _decode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, str) else text.decode("utf-8")

def _encode(text):
    return text if sys.version_info[0] >= 3 or not isinstance(text, unicode) else text.encode("utf-8")

def load_conllu(file):
    class UDRepresentation:
        def __init__(self):
            self.characters = []
            self.tokens = []
            self.words = []
            self.sentences = []
    class UDSpan:
        def __init__(self, start, end):
            self.start = start
            self.end = end
    class UDWord:
        def __init__(self, span, columns, is_multiword):
            self.span = span
            self.columns = columns
            self.is_multiword = is_multiword
            self.parent = None
            self.functional_children = []
            self.columns[FEATS] = "|".join(sorted(feat for feat in columns[FEATS].split("|")
                                                  if feat.split("=", 1)[0] in UNIVERSAL_FEATURES))
            self.columns[DEPREL] = columns[DEPREL].split(":")[0]
            self.is_content_deprel = self.columns[DEPREL] in CONTENT_DEPRELS
            self.is_functional_deprel = self.columns[DEPREL] in FUNCTIONAL_DEPRELS
    ud = UDRepresentation()
    index, sentence_start = 0, None
    while True:
        line = file.readline()
        if not line:
            break
        line = _decode(line.rstrip("\r\n"))
        if sentence_start is None:
            if line.startswith("#"):
                continue
            ud.sentences.append(UDSpan(index, 0))
            sentence_start = len(ud.words)
        if not line:
            def process_word(word):
                if word.parent == "remapping":
                    raise UDError("There is a cycle in a sentence")
                if word.parent is None:
                    head = int(word.columns[HEAD])
                    if head < 0 or head > len(ud.words) - sentence_start:
                        raise UDError("HEAD \"{}\" points outside of the sentence".format(_encode(word.columns[HEAD])))
                    if head:
                        parent = ud.words[sentence_start + head - 1]
                        word.parent = "remapping"
                        process_word(parent)
                        word.parent = parent
            for word in ud.words[sentence_start:]:
                process_word(word)
            for word in ud.words[sentence_start:]:
                if word.parent and word.is_functional_deprel:
                    word.parent.functional_children.append(word)
            if sentence_start < len(ud.words) and len([word for word in ud.words[sentence_start:] if word.parent is None]) != 1:
                raise UDError("There are multiple roots in a sentence")
            ud.sentences[-1].end = index
            sentence_start = None
            continue
        columns = line.split("\t")
        if len(columns) != 10:
            raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: \"{}\"".format(_encode(line)))
        if "." in columns[ID]:
            continue
        columns[FORM] = "".join(filter(lambda c: unicodedata.category(c) != "Zs", columns[FORM]))
        if not columns[FORM]:
            raise UDError("There is an empty FORM in the CoNLL-U file")
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])
        if "-" in columns[ID]:
            try:
                start, end = map(int, columns[ID].split("-"))
            except:
                raise UDError("Cannot parse multi-word token ID \"{}\"".format(_encode(columns[ID])))
            for _ in range(start, end + 1):
                word_line = _decode(file.readline().rstrip("\r\n"))
                word_columns = word_line.split("\t")
                if len(word_columns) != 10:
                    raise UDError("The CoNLL-U line does not contain 10 tab-separated columns: \"{}\"".format(_encode(word_line)))
                ud.words.append(UDWord(ud.tokens[-1], word_columns, is_multiword=True))
        else:
            try:
                word_id = int(columns[ID])
            except:
                raise UDError("Cannot parse word ID \"{}\"".format(_encode(columns[ID])))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID \"{}\" for word \"{}\", expected \"{}\"".format(
                    _encode(columns[ID]), _encode(columns[FORM]), len(ud.words) - sentence_start + 1))
            try:
                head_id = int(columns[HEAD])
            except:
                raise UDError("Cannot parse HEAD \"{}\"".format(_encode(columns[HEAD])))
            if head_id < 0:
                raise UDError("HEAD cannot be negative")
            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))
    if sentence_start is not None:
        raise UDError("The CoNLL-U file does not end with empty line")
    return ud

def sort(packed, ref, reverse=True):
    assert (isinstance(packed, tuple) or isinstance(packed, list)) and isinstance(ref, list)
    packed = [ref] + [range(len(ref))] + list(packed)
    sorted_packed = [list(t) for t in zip(*sorted(zip(*packed), reverse=reverse))]
    return tuple(sorted_packed[1:])

def unsort(sorted_list, oidx):
    assert len(sorted_list) == len(oidx), "Number of list elements must match with original indices."
    _, unsorted = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted

def tensor_unsort(sorted_tensor, oidx):
    assert sorted_tensor.size(0) == len(oidx), "Number of list elements must match with original indices."
    backidx = [x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]
    return sorted_tensor[backidx]