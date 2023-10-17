import os
from pathlib import Path
from TranslatorFromNikita._constants import LEMMA
from TranslatorFromNikita.registry import PIPELINE_NAMES, PROCESSOR_VARIANTS


HOME_DIR = str(Path.home())
DEFAULT_MODEL_DIR = os.getenv("STANZA_RESOURCES_DIR", os.path.join(HOME_DIR, "stanza_resources"))


def build_default_config(resources, lang, dir, load_list):
    default_config = {}
    for item in load_list:
        processor, package, dependencies = item
        if package in PROCESSOR_VARIANTS[processor]:
            default_config[f"{processor}_with_{package}"] = True
        elif processor == LEMMA and package == "identity":
            default_config[f"{LEMMA}_use_identity"] = True
        else:
            default_config[f"{processor}_model_path"] = os.path.join(dir, lang, processor, package + ".pt")
        if not dependencies: 
            continue
        for dependency in dependencies:
            dep_processor, dep_model = dependency
            default_config[f"{processor}_{dep_processor}_path"] = os.path.join(dir, lang, dep_processor, dep_model + ".pt")
    return default_config

def sort_processors(processor_list):
    sorted_list = []
    for processor in PIPELINE_NAMES:
        for item in processor_list:
            if item[0] == processor:
                sorted_list.append(item)
    return sorted_list

def maintain_processor_list(resources, lang, package, processors):
    processor_list = {}
    if processors:
        for key, value in processors.items():
            assert(key in PIPELINE_NAMES)
            assert(isinstance(key, str) and isinstance(value, str))
            if key in resources[lang] and value in resources[lang][key]:
                processor_list[key] = value
            elif key in resources[lang]["default_processors"] and value == "default":
                processor_list[key] = resources[lang]["default_processors"][key]
            elif value in PROCESSOR_VARIANTS[key]:
                processor_list[key] = value
            elif key == LEMMA and value == "identity":
                processor_list[key] = value
            elif key not in resources[lang]:
                processor_list[key] = value
    if package:
        if package == "default":
            for key, value in resources[lang]["default_processors"].items():
                if key not in processor_list:
                    processor_list[key] = value
        else:
            for key in PIPELINE_NAMES:
                if key not in resources[lang]: continue
                if package in resources[lang][key]:
                    if key not in processor_list:
                        processor_list[key] = package
    processor_list = [[key, value] for key, value in processor_list.items()]
    processor_list = sort_processors(processor_list)
    return processor_list

def add_dependencies(resources, lang, processor_list):
    default_dependencies = resources[lang]["default_dependencies"]
    for item in processor_list:
        processor, package = item
        dependencies = default_dependencies.get(processor, None)
        if not any([
                package in PROCESSOR_VARIANTS[processor],
                processor == LEMMA and package == "identity"
            ]):
            dependencies = resources[lang].get(processor, {}).get(package, {}) \
                .get("dependencies", dependencies)
        if dependencies:
            dependencies = [[dependency["model"], dependency["package"]] \
                for dependency in dependencies]
        item.append(dependencies)
    return processor_list

def process_pipeline_parameters(lang, dir, package, processors):
    if isinstance(lang, str):
        lang = lang.strip().lower()
    elif lang is not None:
        raise Exception(
            f"The parameter \"lang\" should be str, "
            f"but got {type(lang).__name__} instead."
        )
    if isinstance(dir, str):
        dir = dir.strip()
    elif dir is not None:
        raise Exception(
            f"The parameter \"dir\" should be str, "
            f"but got {type(dir).__name__} instead."
        )
    if isinstance(package, str):
        package = package.strip().lower()
    elif package is not None:
        raise Exception(
            f"The parameter \"package\" should be str, "
            f"but got {type(package).__name__} instead."
        )
    if isinstance(processors, str):
        processors = {
            processor.strip().lower(): package \
                for processor in processors.split(",")
        }
        package = None
    elif isinstance(processors, dict):
        processors = {
            k.strip().lower(): v.strip().lower() \
                for k, v in processors.items()
        }
    elif processors is not None:
        raise Exception(
            f"The parameter \"processors\" should be dict or str, "
            f"but got {type(processors).__name__} instead."
        )
    return lang, dir, package, processors