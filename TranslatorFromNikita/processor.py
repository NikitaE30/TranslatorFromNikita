from abc import ABC, abstractmethod
from TranslatorFromNikita.registry import NAME_TO_PROCESSOR_CLASS, PIPELINE_NAMES, PROCESSOR_VARIANTS


class ProcessorRequirementsException(Exception):
    def __init__(self, processors_list, err_processor, provided_reqs):
        self._err_processor = err_processor
        self.err_processor.mark_inactive()
        self._processors_list = processors_list
        self._provided_reqs = provided_reqs
        self.build_message()

    @property
    def err_processor(self):
        return self._err_processor

    @property
    def processor_type(self):
        return type(self.err_processor).__name__

    @property
    def processors_list(self):
        return self._processors_list

    @property
    def provided_reqs(self):
        return self._provided_reqs

    def build_message(self):
        prc = ",".join(self.processors_list)
        self.message = (f"---\nPipeline Requirements Error!\n"
                        f"\tProcessor: {self.processor_type}\n"
                        f"\tPipeline processors list: {prc}\n"
                        f"\tProcessor Requirements: {self.err_processor.requires}\n"
                        f"\t\t- fulfilled: {self.err_processor.requires.intersection(self.provided_reqs)}\n"
                        f"\t\t- missing: {self.err_processor.requires - self.provided_reqs}\n"
                        f"\nThe processors list provided for this pipeline is invalid.  Please make sure all "
                        f"prerequisites are met for every processor.\n\n")

    def __str__(self):
        return self.message

class Processor(ABC):
    def __init__(self, config, pipeline, use_gpu):
        self._config = config
        self._pipeline = pipeline
        self._set_up_variants(config, use_gpu)
        self._set_up_requires()
        self._set_up_provides()
        self._check_requirements()
        if hasattr(self, "_variant") and self._variant.OVERRIDE:
            self.process = self._variant.process

    @abstractmethod
    def process(self, doc):
        ...

    def _set_up_provides(self):
        self._provides = self.__class__.PROVIDES_DEFAULT

    def _set_up_requires(self):
        self._requires = self.__class__.REQUIRES_DEFAULT

    def _set_up_variants(self, config, _):
        processor_name = list(self.__class__.PROVIDES_DEFAULT)[0]
        if any(config.get(f"with_{variant}", False) for variant in PROCESSOR_VARIANTS[processor_name]):
            self._trainer = None
            variant_name = [variant for variant in PROCESSOR_VARIANTS[processor_name] if config.get(f"with_{variant}", False)][0]
            self._variant = PROCESSOR_VARIANTS[processor_name][variant_name](config)

    @property
    def config(self):
        return self._config

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def provides(self):
        return self._provides

    @property
    def requires(self):
        return self._requires

    def _check_requirements(self):
        provided_reqs = set.union(*[processor.provides for processor in self.pipeline.loaded_processors]+[set([])])
        if self.requires - provided_reqs:
            load_names = [item[0] for item in self.pipeline.load_list]
            raise ProcessorRequirementsException(load_names, self, provided_reqs)

class ProcessorVariant(ABC):
    OVERRIDE = False

    @abstractmethod
    def process(self, doc):
        ...

class UDProcessor(Processor):
    def __init__(self, config, pipeline, use_gpu):
        super().__init__(config, pipeline, use_gpu)
        self._pretrain = None
        self._trainer = None
        self._vocab = None
        if not hasattr(self, "_variant"):
            self._set_up_model(config, use_gpu)
        self._set_up_final_config(config)

    @abstractmethod
    def _set_up_model(self, config, gpu):
        ...

    def _set_up_final_config(self, config):
        if self._trainer is not None:
            loaded_args, self._vocab = self._trainer.args, self._trainer.vocab
            loaded_args = {k: v for k, v in loaded_args.items() if not UDProcessor.filter_out_option(k)}
        else:
            loaded_args = {}
        loaded_args.update(config)
        self._config = loaded_args

    def mark_inactive(self):
        self._trainer = None
        self._vocab = None

    @property
    def pretrain(self):
        return self._pretrain

    @property
    def trainer(self):
        return self._trainer

    @property
    def vocab(self):
        return self._vocab

    @staticmethod
    def filter_out_option(option):
        options_to_filter = ["cpu", "cuda", "dev_conll_gold", "epochs", "lang", "mode", "save_name", "shorthand"]
        if option.endswith("_file") or option.endswith("_dir"):
            return True
        elif option in options_to_filter:
            return True
        else:
            return False

class ProcessorRegisterException(Exception):
    def __init__(self, processor_class, expected_parent):
        self._processor_class = processor_class
        self._expected_parent = expected_parent
        self.build_message()

    def build_message(self):
        self.message = f"Failed to register \"{self._processor_class}\". It must be a subclass of \"{self._expected_parent}\"."

    def __str__(self):
        return self.message

def register_processor(name):
    def wrapper(Cls):
        if not issubclass(Cls, Processor):
            raise ProcessorRegisterException(Cls, Processor)
        NAME_TO_PROCESSOR_CLASS[name] = Cls
        PIPELINE_NAMES.append(name)
        return Cls
    return wrapper

def register_processor_variant(name, variant):
    def wrapper(Cls):
        if not issubclass(Cls, ProcessorVariant):
            raise ProcessorRegisterException(Cls, ProcessorVariant)
        PROCESSOR_VARIANTS[name][variant] = Cls
        return Cls
    return wrapper