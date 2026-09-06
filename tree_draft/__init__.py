"""Progressive Tree Drafting package.

Public helpers are imported lazily so lightweight commands do not require the
full GPU inference stack at import time.
"""

from .generation import PTDDecodeConfig


def _utils_function(name):
    from . import utils
    return getattr(utils, name)


def augment_all(*args, **kwargs):
    return _utils_function("augment_all")(*args, **kwargs)


def augment_generate(*args, **kwargs):
    return _utils_function("augment_generate")(*args, **kwargs)


def augment_llama(*args, **kwargs):
    return _utils_function("augment_llama")(*args, **kwargs)


def ini_model(*args, **kwargs):
    return _utils_function("ini_model")(*args, **kwargs)


def load_model(*args, **kwargs):
    return _utils_function("load_model")(*args, **kwargs)


def get_inference_id(*args, **kwargs):
    return _utils_function("get_inference_id")(*args, **kwargs)


def start_cache_server(*args, **kwargs):
    return _utils_function("start_cache_server")(*args, **kwargs)


def ensure_cache_server(*args, **kwargs):
    from .cache import ensure_cache_server as _ensure_cache_server
    return _ensure_cache_server(*args, **kwargs)


def stop_cache_server(*args, **kwargs):
    from .cache import stop_cache_server as _stop_cache_server
    return _stop_cache_server(*args, **kwargs)


def enable_ptd(*args, **kwargs):
    from .generation import enable_ptd as _enable_ptd
    return _enable_ptd(*args, **kwargs)


__all__ = [
    "augment_all",
    "augment_generate",
    "augment_llama",
    "ini_model",
    "load_model",
    "get_inference_id",
    "start_cache_server",
    "ensure_cache_server",
    "stop_cache_server",
    "enable_ptd",
    "PTDDecodeConfig",
]
