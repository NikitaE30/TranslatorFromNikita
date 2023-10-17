from translator import doc
from TranslatorFromNikita._constants import TOKENIZE
from TranslatorFromNikita.processor import ProcessorVariant, register_processor_variant

@register_processor_variant(TOKENIZE, "spacy")
class SpacyTokenizer(ProcessorVariant):
    def __init__(self, _):
        ...

    def process(self, text):
        if not isinstance(text, str):
            raise Exception("Must supply a string to the spaCy tokenizer.")
        spacy_doc = self.nlp(text)
        sentences = []
        for sent in spacy_doc.sents:
            tokens = []
            for tok in sent:
                token_entry = {
                    doc.TEXT: tok.text,
                    doc.MISC: f"{doc.START_CHAR}={tok.idx}|{doc.END_CHAR}={tok.idx+len(tok.text)}"
                }
                tokens.append(token_entry)
            sentences.append(tokens)
        return doc.Document(sentences, text)