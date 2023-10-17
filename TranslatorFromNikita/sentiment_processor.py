from TranslatorFromNikita._constants import *
from TranslatorFromNikita.processor import UDProcessor, register_processor


@register_processor(SENTIMENT)
class SentimentProcessor(UDProcessor):
    PROVIDES_DEFAULT = set([SENTIMENT])
    REQUIRES_DEFAULT = set([TOKENIZE])