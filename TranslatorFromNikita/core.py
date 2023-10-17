import io
import torch
import json
import os
from TranslatorFromNikita._constants import *
from TranslatorFromNikita.doc import Document
from TranslatorFromNikita.processor import ProcessorRequirementsException
from TranslatorFromNikita.registry import NAME_TO_PROCESSOR_CLASS, PIPELINE_NAMES
from TranslatorFromNikita.tokenize_processor import TokenizeProcessor
from TranslatorFromNikita.mwt_processor import MWTProcessor
from TranslatorFromNikita.pos_processor import POSProcessor
from TranslatorFromNikita.lemma_processor import LemmaProcessor
from TranslatorFromNikita.depparse_processor import DepparseProcessor
from TranslatorFromNikita.sentiment_processor import SentimentProcessor
from TranslatorFromNikita.ner_processor import NERProcessor
from TranslatorFromNikita.common import (DEFAULT_MODEL_DIR, 
                                        maintain_processor_list, 
                                        add_dependencies, 
                                        build_default_config, 
                                        process_pipeline_parameters, 
                                        sort_processors)


class PipelineRequirementsException(Exception):
    def __init__(self, processor_req_fails):
        self._processor_req_fails = processor_req_fails
        self.build_message()

    @property
    def processor_req_fails(self):
        return self._processor_req_fails

    def build_message(self):
        err_msg = io.StringIO()
        print(*[req_fail.message for req_fail in self.processor_req_fails], sep="\n", file=err_msg)
        self.message = "\n\n" + err_msg.getvalue()

    def __str__(self):
        return self.message

class Pipeline:
    def __init__(self, lang="en", dir=DEFAULT_MODEL_DIR, package="default", processors={}, logging_level="INFO", verbose=None, use_gpu=True, **kwargs):
        self.lang, self.dir, self.kwargs = lang, dir, kwargs
        lang, dir, package, processors = process_pipeline_parameters(lang, dir, package, processors)
        resources_filepath = os.path.join(dir, "resources.json")
        if not os.path.exists(resources_filepath):
            raise Exception(f"Resources file not found at: {resources_filepath}. Try to download the model again.")
        with open(resources_filepath) as infile:
            resources = json.load(infile)
        if lang in resources:
            if "alias" in resources[lang]:
                lang = resources[lang]["alias"]
        self.load_list = maintain_processor_list(resources, lang, package, processors) if lang in resources else []
        self.load_list = add_dependencies(resources, lang, self.load_list) if lang in resources else []
        self.load_list = self.update_kwargs(kwargs, self.load_list)
        if len(self.load_list) == 0: raise Exception("No processor to load. Please check if your language or package is correctly set.")
        self.config = build_default_config(resources, lang, dir, self.load_list)
        self.config.update(kwargs)
        self.processors = {}
        pipeline_level_configs = {"lang": lang, "mode": "predict"}
        self.use_gpu = torch.cuda.is_available() and use_gpu
        pipeline_reqs_exceptions = []
        for item in self.load_list:
            processor_name, _, _ = item
            curr_processor_config = self.filter_config(processor_name, self.config)
            curr_processor_config.update(pipeline_level_configs)
            try:
                self.processors[processor_name] = NAME_TO_PROCESSOR_CLASS[processor_name](config=curr_processor_config,
                                                                                          pipeline=self,
                                                                                          use_gpu=self.use_gpu)
            except ProcessorRequirementsException as e:
                pipeline_reqs_exceptions.append(e)
                self.processors[processor_name] = e.err_processor
        if pipeline_reqs_exceptions:
            raise PipelineRequirementsException(pipeline_reqs_exceptions)

    def update_kwargs(self, kwargs, processor_list):
        processor_dict = {processor: {"package": package, "dependencies": dependencies} for (processor, package, dependencies) in processor_list}
        for key, value in kwargs.items():
            k, v = key.split("_", 1)
            if v == "model_path":
                package = value if len(value) < 25 else value[:10]+ "..." + value[-10:]
                dependencies = processor_dict.get(k, {}).get("dependencies")
                processor_dict[k] = {"package": package, "dependencies": dependencies}
        processor_list = [[processor, processor_dict[processor]["package"], processor_dict[processor]["dependencies"]] for processor in processor_dict]
        processor_list = sort_processors(processor_list)
        return processor_list

    def filter_config(self, prefix, config_dict):
        filtered_dict = {}
        for key in config_dict.keys():
            k, v = key.split("_", 1)
            if k == prefix:
                filtered_dict[v] = config_dict[key]
        return filtered_dict

    @property
    def loaded_processors(self):
        return [self.processors[processor_name] for processor_name in PIPELINE_NAMES if self.processors.get(processor_name)]

    def process(self, doc):
        for processor_name in PIPELINE_NAMES:
            if self.processors.get(processor_name):
                doc = self.processors[processor_name].process(doc)
        return doc

    def __call__(self, doc):
        assert any([isinstance(doc, str), isinstance(doc, list),
                    isinstance(doc, Document)]), "input should be either str, list or Document"
        doc = self.process(doc)
        return doc