import importlib
import logging

logger = logging.getLogger(__name__)


def lazy_import(module_name: str):
    logger.debug("Loading module %s", module_name)
    return importlib.import_module(module_name)


def safe_lazy_import(module_name: str):
    """Import a module and log a helpful error if it is missing."""
    try:
        return lazy_import(module_name)
    except ModuleNotFoundError:
        pkg = module_name.split(".")[0]
        logger.error(
            "The '%s' package is not installed. Please run 'pip install %s' to enable functionality.",
            pkg,
            pkg,
            exc_info=True,
        )
        raise
