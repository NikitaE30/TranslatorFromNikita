from translator import doc
from TranslatorFromNikita._constants import TOKENIZE
from TranslatorFromNikita.processor import ProcessorVariant, register_processor_variant

@register_processor_variant(TOKENIZE, "sudachipy")
class SudachiPyTokenizer(ProcessorVariant):
    def __init__(self, _):
        ...

    def process(self, text):
        if not isinstance(text, str):
            raise Exception("Must supply a string to the SudachiPy tokenizer.")
        tokens = self.tokenizer.tokenize(text)
        sentences = []
        current_sentence = []
        for token in tokens:
            token_text = token.surface()
            if token_text.isspace():
                continue
            start = token.begin()
            end = token.end()
            token_entry = {
                doc.TEXT: token_text,
                doc.MISC: f"{doc.START_CHAR}={start}|{doc.END_CHAR}={end}"
            }
            current_sentence.append(token_entry)
            if token_text in ["。", "！", "？", "!", "?"]:
                sentences.append(current_sentence)
                current_sentence = []
        if len(current_sentence) > 0:
            sentences.append(current_sentence)
        return doc.Document(sentences, text)