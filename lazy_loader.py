import importlib
import logging

logger = logging.getLogger(__name__)

def lazy_import(module_name: str):
    logger.debug("Loading module %s", module_name)
    return importlib.import_module(module_name)
