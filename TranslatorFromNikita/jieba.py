import re
from translator import doc
from TranslatorFromNikita._constants import TOKENIZE
from TranslatorFromNikita.processor import ProcessorVariant, register_processor_variant


@register_processor_variant(TOKENIZE, "jieba")
class JiebaTokenizer(ProcessorVariant):
    def __init__(self, config):
        ...

    def process(self, text):
        if not isinstance(text, str):
            raise Exception("Must supply a string to the Jieba tokenizer.")
        tokens = self.nlp.cut(text, cut_all=False)
        sentences = []
        current_sentence = []
        offset = 0
        for token in tokens:
            if re.match(r"\s+", token):
                offset += len(token)
                continue
            token_entry = {
                doc.TEXT: token,
                doc.MISC: f"{doc.START_CHAR}={offset}|{doc.END_CHAR}={offset+len(token)}"
            }
            current_sentence.append(token_entry)
            offset += len(token)
            if token in ["。", "！", "？", "!", "?"]:
                sentences.append(current_sentence)
                current_sentence = []
        if len(current_sentence) > 0:
            sentences.append(current_sentence)
        return doc.Document(sentences, text)