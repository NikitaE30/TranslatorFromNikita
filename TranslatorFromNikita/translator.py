from __future__ import annotations
import os
import copy
from pathlib import Path
import json
from typing import List
import sentencepiece as spm
from ctranslate2 import Translator
from difflib import SequenceMatcher
import zipfile
from TranslatorFromNikita.core import Pipeline


TRUE_VALUES = ["1", "TRUE", "True", "true"]
stanza_available = os.getenv("ARGOS_STANZA_AVAILABLE") in (TRUE_VALUES + [None])
installed_translates: List[InstalledTranslate] = []
device = os.environ.get("ARGOS_DEVICE_TYPE", "cpu")
openai_api_key = os.getenv("OPENAI_API_KEY", None)
fewshot_prompt = """<detect-sentence-boundaries> I walked down to the river. Then I went to the
I walked down to the river. <sentence-boundary>
----------
<detect-sentence-boundaries> Argos Translate is machine translation software. It is also
Argos Translate is machine translation software. <sentence-boundary>
----------
<detect-sentence-boundaries> Argos Translate is written in Python and uses OpenAI. It also supports
Argos Translate is written in Python and uses OpenAI. <sentence-boundary>
----------
"""
FEWSHOT_BOUNDARY_TOKEN = "-" * 10
prompt = """Translate to French (fr)
From English (es)
==========
Bramshott is a village with mediaeval origins in the East Hampshire district of Hampshire, England. It lies 0.9 miles (1.4 km) north of Liphook. The nearest railway station, Liphook, is 1.3 miles (2.1 km) south of the village. 
----------
Bramshott est un village avec des origines médiévales dans le quartier East Hampshire de Hampshire, en Angleterre. Il se trouve à 0,9 miles (1,4 km) au nord de Liphook. La gare la plus proche, Liphook, est à 1,3 km (2,1 km) au sud du village.
==========

Translate to Russian (rs)
From German (de)
==========
Der Gewöhnliche Strandhafer (Ammophila arenaria (L.) Link; Syn: Calamagrostis arenaria (L.) Roth) – auch als Gemeiner Strandhafer, Sandrohr, Sandhalm, Seehafer oder Helm (niederdeutsch) bezeichnet – ist eine zur Familie der Süßgräser (Poaceae) gehörige Pionierpflanze. 
----------
Обычная пляжная овсянка (аммофила ареалия (л.) соединение; сина: каламагростисная анария (л.) Рот, также называемая обычной пляжной овцой, песчаной, сандалмой, морской орой или шлемом (нижний немецкий) - это кукольная станция, принадлежащая семье сладких трав (поа).
==========
"""
LEMMA = "lemma"


class Hypothesis:
    value: str
    score: float

    def __init__(self, value: str, score: float):
        self.value = value
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"({repr(self.value)}, {self.score})"

    def __str__(self):
        return repr(self)

class ITranslation:
    from_lang: Language
    to_lang: Language

    def translate(self, input_text: str) -> str:
        return self.hypotheses(input_text, num_hypotheses=1)[0].value

    def hypotheses(self, input_text: str, num_hypotheses: int = 4) -> list[Hypothesis]:
        raise NotImplementedError()

    @staticmethod
    def split_into_paragraphs(input_text: str) -> list[str]:
        return input_text.split("\n")

    @staticmethod
    def combine_paragraphs(paragraphs: list[str]) -> str:
        return "\n".join(paragraphs)

    def __repr__(self):
        return str(self.from_lang) + " -> " + str(self.to_lang)

    def __str__(self):
        return repr(self).replace("->", "→")

class Language:
    translations_from: list[ITranslation] = []
    translations_to: list[ITranslation] = []

    def __init__(self, code: str, name: str):
        self.code = code
        self.name = name
        self.translations_from = []
        self.translations_to = []

    def __str__(self):
        return self.name

    def get_translation(self, to: Language) -> ITranslation | None:
        valid_translations = list(
            filter(lambda x: x.to_lang.code == to.code, self.translations_from)
        )
        if len(valid_translations) > 0:
            return valid_translations[0]
        return None

class IPackage:
    code: str
    package_path: Path
    package_version: str
    argos_version: str
    from_code: str
    from_name: str
    from_codes: list
    to_code: str
    to_codes: list
    to_name: str
    links: list
    type: str
    languages: list
    dependencies: list
    source_languages: list
    target_languages: list
    links: list[str]

    def load_metadata_from_json(self, metadata):
        self.code = metadata.get("code")
        self.package_version = metadata.get("package_version", "")
        self.argos_version = metadata.get("argos_version", "")
        self.from_code = metadata.get("from_code")
        self.from_name = metadata.get("from_name", "")
        self.from_codes = metadata.get("from_codes", list())
        self.to_code = metadata.get("to_code")
        self.to_codes = metadata.get("to_codes", list())
        self.to_name = metadata.get("to_name", "")
        self.links = metadata.get("links", list())
        self.type = metadata.get("type", "translate")
        self.languages = metadata.get("languages", list())
        self.dependencies = metadata.get("dependencies", list())
        self.source_languages = metadata.get("source_languages", list())
        self.target_languages = metadata.get("target_languages", list())
        if self.from_code is not None or self.from_name is not None:
            from_lang = dict()
            if self.from_code is not None:
                from_lang["code"] = self.from_code
            if self.from_name is not None:
                from_lang["name"] = self.from_name
            self.source_languages.append(from_lang)
        if self.to_code is not None or self.to_name is not None:
            to_lang = dict()
            if self.to_code is not None:
                to_lang["code"] = self.to_code
            if self.to_name is not None:
                to_lang["name"] = self.to_name
            self.source_languages.append(to_lang)
        self.source_languages += copy.deepcopy(self.languages)
        self.target_languages += copy.deepcopy(self.languages)

    def get_readme(self) -> str:
        raise NotImplementedError()

    def get_description(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return (
            self.package_version == other.package_version
            and self.argos_version == other.argos_version
            and self.from_code == other.from_code
            and self.from_name == other.from_name
            and self.to_code == other.to_code
            and self.to_name == other.to_name
        )

    def __repr__(self):
        if len(self.from_name) > 0 and len(self.to_name) > 0:
            return "{} -> {}".format(self.from_name, self.to_name)
        elif self.type:
            return self.type
        return ""

    def __str__(self):
        return repr(self).replace("->", "→")

class Package(IPackage):
    def __init__(self, package_path: Path):
        if type(package_path) == str:
            package_path = Path(package_path)
        self.package_path = package_path
        metadata_path = package_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                "Error opening package at " + str(metadata_path) + " no metadata.json"
            )
        with open(metadata_path) as metadata_file:
            metadata = json.load(metadata_file)
            self.load_metadata_from_json(metadata)

    def get_readme(self) -> str | None:
        readme_path = self.package_path / "README.md"
        if not readme_path.exists():
            return None
        with open(readme_path, "r") as readme_file:
            return readme_file.read()

    def get_description(self):
        return self.get_readme()

class CachedTranslation(ITranslation):
    underlying: ITranslation
    from_lang: Language
    to_lang: Language
    cache: dict

    def __init__(self, underlying: ITranslation):
        self.underlying = underlying
        self.from_lang = underlying.from_lang
        self.to_lang = underlying.to_lang
        self.cache = dict()

    def hypotheses(self, input_text: str, num_hypotheses: int = 4) -> list[Hypothesis]:
        new_cache = dict()
        paragraphs = ITranslation.split_into_paragraphs(input_text)
        translated_paragraphs = []
        for paragraph in paragraphs:
            translated_paragraph = self.cache.get(paragraph)
            if (
                translated_paragraph is None
                or len(translated_paragraph) != num_hypotheses
            ):
                translated_paragraph = self.underlying.hypotheses(
                    paragraph, num_hypotheses
                )
            new_cache[paragraph] = translated_paragraph
            translated_paragraphs.append(translated_paragraph)
        self.cache = new_cache
        hypotheses_to_return = [Hypothesis("", 0) for i in range(num_hypotheses)]
        for i in range(num_hypotheses):
            for j in range(len(translated_paragraphs)):
                value = ITranslation.combine_paragraphs(
                    [hypotheses_to_return[i].value, translated_paragraphs[j][i].value]
                )
                score = (
                    hypotheses_to_return[i].score + translated_paragraphs[j][i].score
                )
                hypotheses_to_return[i] = Hypothesis(value, score)
            hypotheses_to_return[i].value = hypotheses_to_return[i].value.lstrip("\n")
        return hypotheses_to_return

class InstalledTranslate:
    package_key: str
    cached_translation: CachedTranslation

class PackageTranslation(ITranslation):
    def __init__(self, from_lang: Language, to_lang: Language, pkg: Package):
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.pkg = pkg
        self.translator = None

    def hypotheses(self, input_text: str, num_hypotheses: int = 4) -> list[Hypothesis]:
        if self.translator is None:
            model_path = str(self.pkg.package_path / "model")
            self.translator = Translator(model_path, device=device)
        paragraphs = ITranslation.split_into_paragraphs(input_text)
        translated_paragraphs = []
        for paragraph in paragraphs:
            translated_paragraphs.append(
                apply_packaged_translation(
                    self.pkg, paragraph, self.translator, num_hypotheses
                )
            )
        hypotheses_to_return = [Hypothesis("", 0) for i in range(num_hypotheses)]
        for i in range(num_hypotheses):
            for translated_paragraph in translated_paragraphs:
                value = ITranslation.combine_paragraphs(
                    [hypotheses_to_return[i].value, translated_paragraph[i].value]
                )
                score = hypotheses_to_return[i].score + translated_paragraph[i].score
                hypotheses_to_return[i] = Hypothesis(value, score)
            hypotheses_to_return[i].value = hypotheses_to_return[i].value.lstrip("\n")
        return hypotheses_to_return

class IdentityTranslation(ITranslation):
    def __init__(self, lang: Language):
        self.from_lang = lang
        self.to_lang = lang

    def hypotheses(self, input_text: str, num_hypotheses: int = 4):
        return [Hypothesis(input_text, 0) for i in range(num_hypotheses)]

class CompositeTranslation(ITranslation):
    t1: ITranslation
    t2: ITranslation
    from_lang: Language
    to_lang: Language

    def __init__(self, t1: ITranslation, t2: ITranslation):
        self.t1 = t1
        self.t2 = t2
        self.from_lang = t1.from_lang
        self.to_lang = t2.to_lang

    def hypotheses(self, input_text: str, num_hypotheses: int = 4) -> list[Hypothesis]:
        t1_hypotheses = self.t1.hypotheses(input_text, num_hypotheses)
        to_return = []
        for t1_hypothesis in t1_hypotheses:
            t2_hypotheses = self.t2.hypotheses(t1_hypothesis.value, num_hypotheses)
            for t2_hypothesis in t2_hypotheses:
                to_return.append(
                    Hypothesis(
                        t2_hypothesis.value, t1_hypothesis.score + t2_hypothesis.score
                    )
                )
        to_return.sort(reverse=True)
        return to_return[0:num_hypotheses]

class ILanguageModel:
    def infer(self, x: str) -> str | None:
        raise NotImplementedError()

def detect_sentence(input_text: str, sbd_translation, sentence_guess_length: int = 150) -> int:
    sentence_guess = input_text[:sentence_guess_length]
    sbd_translated_guess = sbd_translation.translate(
        "<detect-sentence-boundaries>" + sentence_guess
    )
    return process_seq2seq_sbd(input_text, sbd_translated_guess)

def get_sbd_package() -> Package | None:
    packages = []
    for file in os.scandir("./TranslatorFromNikita/translators/"):
        if os.path.isfile(file.path):
            continue
        packages.append(Package(file.path))
    for pkg in packages:
        if pkg.type == "sbd":
            return pkg
    return None

def apply_packaged_translation(
    pkg: Package, input_text: str, translator: Translator, num_hypotheses: int = 4) -> list[Hypothesis]:
    if pkg.type == "sbd":
        sentences = [input_text]
    elif stanza_available:
        stanza_pipeline = Pipeline(
            lang=pkg.from_code,
            dir=str(pkg.package_path / "stanza"),
            processors="tokenize",
            use_gpu=device == "cuda",
            logging_level="WARNING",
        )
        stanza_sbd = stanza_pipeline(input_text)
        sentences = [sentence.text for sentence in stanza_sbd.sentences]
    else:
        DEFAULT_SENTENCE_LENGTH = 250
        sentences = []
        start_index = 0
        sbd_package = get_sbd_package()
        assert sbd_package is not None
        sbd_translation = PackageTranslation(None, None, sbd_package)
        while start_index < len(input_text) - 1:
            detected_sentence_index = detect_sentence(
                input_text[start_index:], sbd_translation
            )
            if detected_sentence_index == -1:
                sbd_index = start_index + DEFAULT_SENTENCE_LENGTH
            else:
                sbd_index = start_index + detected_sentence_index
            sentences.append(input_text[start_index:sbd_index])
            start_index = sbd_index
    sp_model_path = str(pkg.package_path / "sentencepiece.model")
    sp_processor = spm.SentencePieceProcessor(model_file=sp_model_path)
    tokenized = [sp_processor.encode(sentence, out_type=str) for sentence in sentences]
    BATCH_SIZE = 32
    translated_batches = translator.translate_batch(
        tokenized,
        replace_unknowns=True,
        max_batch_size=BATCH_SIZE,
        beam_size=max(num_hypotheses, 4),
        num_hypotheses=num_hypotheses,
        length_penalty=0.2,
        return_scores=True,
    )
    value_hypotheses = []
    for i in range(num_hypotheses):
        translated_tokens = []
        cumulative_score = 0
        for translated_batch in translated_batches:
            translated_tokens += translated_batch[i]["tokens"]
            cumulative_score += translated_batch[i]["score"]
        detokenized = "".join(translated_tokens)
        detokenized = detokenized.replace("▁", " ")
        value = detokenized
        if len(value) > 0 and value[0] == " ":
            value = value[1:]
        hypothesis = Hypothesis(value, cumulative_score)
        value_hypotheses.append(hypothesis)
    return value_hypotheses

def process_seq2seq_sbd(input_text: str, sbd_translated_guess: str) -> int:
    sbd_translated_guess_index = sbd_translated_guess.find("<sentence-boundary>")
    if sbd_translated_guess_index != -1:
        sbd_translated_guess = sbd_translated_guess[:sbd_translated_guess_index]
        best_index = None
        best_ratio = 0.0
        for i in range(len(input_text)):
            candidate_sentence = input_text[:i]
            sm = SequenceMatcher()
            sm.set_seqs(candidate_sentence, sbd_translated_guess)
            ratio = sm.ratio()
            if best_index is None or ratio > best_ratio:
                best_index = i
                best_ratio = ratio
        return best_index
    else:
        return -1

def get_installed_languages() -> list[Language]:
    packages = []
    for file in os.scandir("./TranslatorFromNikita/translators/"):
        if os.path.isfile(file.path):
            continue
        packages.append(Package(file.path))
    if not stanza_available:
        sbd_packages = list(filter(lambda x: x.type == "sbd", packages))
        sbd_available_codes = set()
        for sbd_package in sbd_packages:
            sbd_available_codes = sbd_available_codes.union(sbd_package.from_codes)
        packages = list(
            filter(lambda x: x.from_code in sbd_available_codes, packages)
        )
    packages = list(filter(lambda x: x.type == "translate", packages))
    language_of_code = dict()
    for pkg in packages:
        if pkg.from_code not in language_of_code:
            language_of_code[pkg.from_code] = Language(pkg.from_code, pkg.from_name)
        if pkg.to_code not in language_of_code:
            language_of_code[pkg.to_code] = Language(pkg.to_code, pkg.to_name)
        from_lang = language_of_code[pkg.from_code]
        to_lang = language_of_code[pkg.to_code]
        package_key = f"{pkg.from_code}-{pkg.to_code}"
        contain = list(
            filter(lambda x: x.package_key == package_key, installed_translates)
        )
        translation_to_add: CachedTranslation
        if len(contain) == 0:
            translation_to_add = CachedTranslation(
                PackageTranslation(from_lang, to_lang, pkg)
            )
            saved_cache = InstalledTranslate()
            saved_cache.package_key = package_key
            saved_cache.cached_translation = translation_to_add
            installed_translates.append(saved_cache)
        else:
            translation_to_add = contain[0].cached_translation
        from_lang.translations_from.append(translation_to_add)
        to_lang.translations_to.append(translation_to_add)
    languages = list(language_of_code.values())
    for language in languages:
        identity_translation = IdentityTranslation(language)
        language.translations_from.append(identity_translation)
        language.translations_to.append(identity_translation)
    for language in languages:
        keep_adding_translations = True
        while keep_adding_translations:
            keep_adding_translations = False
            for translation in language.translations_from:
                for translation_2 in translation.to_lang.translations_from:
                    if language.get_translation(translation_2.to_lang) is None:
                        keep_adding_translations = True
                        composite_translation = CompositeTranslation(
                            translation, translation_2
                        )
                        language.translations_from.append(composite_translation)
                        translation_2.to_lang.translations_to.append(
                            composite_translation
                        )
    en_index = None
    for i, language in enumerate(languages):
        if language.code == "en":
            en_index = i
            break
    english = None
    if en_index is not None:
        english = languages.pop(en_index)
    languages.sort(key=lambda x: x.name)
    if english is not None:
        languages = [english] + languages
    return languages

def get_language_from_code(code: str) -> Language | None:
    return next(filter(lambda x: x.code == code, get_installed_languages()), None)

def get_translation_from_codes(from_code: str, to_code: str) -> ITranslation:
    from_lang = get_language_from_code(from_code)
    to_lang = get_language_from_code(to_code)
    return from_lang.get_translation(to_lang)

def translate(text: str, from_code: str, to_code: str) -> str:
    """
    A function designed for direct interaction with the user. 
    It is used to translate text from one language to another.
    Arguments:
        text (str): text to translate.
        from_code (str): the code of the language you want to translate from.
        to_code (str): the code of the language you want to translate into.
    Returns:
        str: translated text.
    """
    translation = get_translation_from_codes(from_code, to_code)
    return translation.translate(text)

def install_from_path(path: str) -> bool:
    """
    A function designed for direct interaction with the user.
    Used to install language models.
    Arguments:
        path (str): the path to the language model.
    Returns:
        bool: notification of completion.
    """
    with zipfile.ZipFile(path, "r") as zipf:
        zipf.extractall(path="./TranslatorFromNikita/translators/")
    return True