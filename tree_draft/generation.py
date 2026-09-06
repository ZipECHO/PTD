"""GenerationMixin-style PTD entry points.

This module intentionally does not patch ``transformers.GenerationMixin``.
Instead, :func:`enable_ptd` attaches an instance-local ``ptd_generate`` method
to a model.  The implementation reuses the existing PTD search loop today and
provides the stable seam that can later be wired into a Transformers fork as a
first-class generation mode.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Mapping, Optional


@dataclass
class PTDDecodeConfig:
    """Configuration for the PTD generation strategy.

    The fields mirror the tree section of the repository's configuration file
    so callers can use this interface without constructing a legacy ``args``
    namespace.
    """

    ini_max_depth: int = 2
    max_depth: int = 6
    exclude_root: int = 1
    max_child_num: list = field(default_factory=lambda: [4] * 10)
    min_child_num: list = field(default_factory=lambda: [4] + [0] * 9)
    default_max_child_num: int = 4
    default_min_child_num: int = 4
    update_sub_nodes: bool = False
    ignore_exceed: bool = False
    tree_update_method: str = "argmax"
    root_visible: bool = False

    def to_tree_config(self) -> Dict[str, Any]:
        """Convert the public config to the legacy tree-config structure."""
        return {
            "ini_max_depth": self.ini_max_depth,
            "max_depth": self.max_depth,
            "exclude_root": self.exclude_root,
            "node_config": {
                "max_child_num": list(self.max_child_num),
                "min_child_num": list(self.min_child_num),
                "default_max_child_num": self.default_max_child_num,
                "default_min_child_num": self.default_min_child_num,
                "update_sub_nodes": self.update_sub_nodes,
                "ignore_exceed": self.ignore_exceed,
            },
            "tree_config": {"root_visible": self.root_visible},
            "tree_update_method": self.tree_update_method,
        }


def _coerce_ptd_config(config: Optional[Any]) -> PTDDecodeConfig:
    if config is None:
        return PTDDecodeConfig()
    if isinstance(config, PTDDecodeConfig):
        return config
    if isinstance(config, Mapping):
        # Accept either the public flat form or the repository's nested tree
        # config, making migration from configs/tree_config.json straightforward.
        values = dict(config)
        node = values.pop("node_config", {}) or {}
        tree = values.pop("tree_config", {}) or {}
        values.update(node)
        if "root_visible" not in values:
            values["root_visible"] = tree.get("root_visible", False)
        return PTDDecodeConfig(**{
            key: value for key, value in values.items()
            if key in PTDDecodeConfig.__dataclass_fields__
        })
    raise TypeError("ptd_config must be PTDDecodeConfig, a mapping, or None")


class PTDGenerationMixin:
    """GenerationMixin-shaped interface for progressive tree drafting.

    ``ptd_generate`` accepts already-tokenized ``input_ids`` and the same
    low-level generation arguments used by Transformers' ``_sample`` method.
    A later integration can call ``_ptd_search`` directly after Transformers'
    normal input/configuration preparation phase.
    """

    def ptd_generate(
        self,
        input_ids=None,
        *,
        ptd_config: Optional[Any] = None,
        generation_config=None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """Generate with PTD without modifying global Transformers classes.

        The model must expose ``tokenizer`` and ``input_ids`` should be a
        tensor produced by the model's tokenizer or processor. The PTD cache
        server is started on demand and cleaned up after this call.
        """
        if input_ids is None:
            raise ValueError("ptd_generate requires tokenized input_ids")

        config = _coerce_ptd_config(ptd_config)
        if not hasattr(self, "generation_config") and generation_config is None:
            raise RuntimeError("The model must provide generation_config")

        if generation_config is None:
            generation_config = deepcopy(self.generation_config)
        else:
            generation_config = deepcopy(generation_config)
        if temperature is not None:
            generation_config.temperature = temperature
        elif getattr(generation_config, "temperature", None) is None:
            generation_config.temperature = 0.0
        if max_new_tokens is None:
            max_new_tokens = getattr(generation_config, "max_new_tokens", None)
        if max_new_tokens is None:
            raise ValueError("ptd_generate requires max_new_tokens")

        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = bool(
            getattr(generation_config, "temperature", 0.0) > 1e-4
        )

        # ``tree_draft_greedy_search`` updates the attention mask after every
        # accepted path.  GenerationMixin normally creates this mask before
        # entering ``_sample``; this standalone entry point must do it itself.
        if "attention_mask" not in kwargs:
            pad_token_id = getattr(generation_config, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(getattr(self, "tokenizer", None), "pad_token_id", None)
            kwargs["attention_mask"] = (
                input_ids.ne(pad_token_id).long() if pad_token_id is not None
                else input_ids.new_ones(input_ids.shape, dtype=input_ids.dtype)
            )

        # PTD's existing loop reads these attributes for compatibility with
        # the legacy main.py path. They are set on this model instance only.
        self.tree_config = config.to_tree_config()
        self.draft_config = SimpleNamespace(
            run_mode="draft",
            model_arch=getattr(self.config, "_name_or_path", self.__class__.__name__),
            inference_id="ptd",
            dist_workers=1,
            use_flash=0,
        )

        # Import runtime-heavy modules lazily so ``python main.py --help`` and
        # configuration inspection remain usable without torch/Transformers.
        from transformers.generation import StoppingCriteriaList
        from transformers.generation.stopping_criteria import MaxLengthCriteria

        stopping_criteria = kwargs.pop("stopping_criteria", None)
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList([
                MaxLengthCriteria(
                    max_length=input_ids.shape[-1] + max_new_tokens,
                    max_position_embeddings=getattr(self.config, "max_position_embeddings", None),
                )
            ])

        kwargs["generation_config"] = generation_config
        kwargs["stopping_criteria"] = stopping_criteria
        from .cache import ensure_cache_server, stop_cache_server

        cache_server = ensure_cache_server(
            batch_size=input_ids.shape[0],
            rest_corpus_path=kwargs.pop("rest_corpus_path", None),
            tokenizer_path=kwargs.pop("tokenizer_path", None),
        )
        try:
            return self._ptd_search(input_ids=input_ids, **kwargs)
        finally:
            if cache_server.auto_cleanup:
                stop_cache_server(cache_server)

    def _ptd_search(self, input_ids, **kwargs):
        """Low-level search hook matching Transformers' generation style."""
        from .decode_strategy.progressive_tree_drafting import tree_draft_greedy_search

        return tree_draft_greedy_search(self, input_ids=input_ids, **kwargs)


def _install_generate_dispatch(model):
    """Install an instance-local ``generate(use_ptd=True)`` dispatch.

    This is deliberately attached to one model instance instead of modifying
    ``transformers.GenerationMixin`` globally.  It gives applications the
    same opt-in API that a future Transformers fork can expose while keeping
    the standard generation path untouched.
    """
    original_generate = getattr(model, "generate", None)
    if not callable(original_generate):
        return
    if getattr(model, "_ptd_generate_dispatch_installed", False):
        return

    # Keep a reference for debugging and for a clean opt-out/restore path.
    model._ptd_original_generate = original_generate

    @wraps(original_generate)
    def generate_with_ptd(*args, use_ptd=False, ptd_config=None, **kwargs):
        if use_ptd:
            if args:
                if "input_ids" in kwargs:
                    raise TypeError("input_ids was provided both positionally and by keyword")
                input_ids, *rest = args
                if rest:
                    raise TypeError(
                        "PTD generation accepts only input_ids as a positional argument; "
                        "pass other arguments by keyword"
                    )
            else:
                input_ids = kwargs.pop("input_ids", None)
            return model.ptd_generate(
                input_ids=input_ids,
                ptd_config=ptd_config,
                **kwargs,
            )

        if ptd_config is not None:
            raise ValueError("ptd_config requires use_ptd=True")
        return original_generate(*args, **kwargs)

    model.generate = generate_with_ptd
    model._ptd_generate_dispatch_installed = True


def enable_ptd(model):
    """Attach PTD methods to one model instance and return that model.

    Unlike the legacy ``augment_generate`` path, this function does not mutate
    ``GenerationMixin`` or any Transformers model class globally.
    """
    model.ptd_generate = MethodType(PTDGenerationMixin.ptd_generate, model)
    model._ptd_search = MethodType(PTDGenerationMixin._ptd_search, model)
    _install_generate_dispatch(model)

    # Bind the existing architecture implementations to this instance only.
    # This is the first step away from the legacy global ``augment_all`` patch.
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type in {"llama", "llama2"}:
        from .models.Llama import llama_ptd
        from .models import ptd_common

        model.PTDforward = MethodType(llama_ptd.PTDforward, model)
        model.model.LlamaModelPTDforward = MethodType(
            llama_ptd.LlamaModelPTDforward, model.model
        )
        model.model.prepare_decoder_attention_mask = MethodType(
            ptd_common.prepare_decoder_attention_mask_batch, model.model
        )
        model.ptd_backend = "llama"
    elif model_type == "qwen2":
        from .models.Qwen import qwen_ptd
        from .models import ptd_common

        model.PTDforward = MethodType(qwen_ptd.PTDfroward, model)
        model.model.QwenModelPTDForward = MethodType(
            qwen_ptd.QwenModelPTDForward, model.model
        )
        model.model.prepare_decoder_attention_mask = MethodType(
            ptd_common.prepare_decoder_attention_mask_batch, model.model
        )
        model.ptd_backend = "qwen2"
    elif model_type == "qwen3":
        from .models.Qwen3 import qwen3_ptd
        from .models import ptd_common

        model.PTDforward = MethodType(qwen3_ptd.PTDfroward, model)
        model.model.QwenModelPTDForward = MethodType(
            qwen3_ptd.QwenModelPTDForward, model.model
        )
        model.model.prepare_decoder_attention_mask = MethodType(
            ptd_common.prepare_decoder_attention_mask_batch, model.model
        )
        model.ptd_backend = "qwen3"
    elif model_type:
        raise ValueError(
            f"PTD backend is not implemented for model_type={model_type!r}. "
            "Supported types are llama, qwen2, and qwen3."
        )
    return model


__all__ = ["PTDDecodeConfig", "PTDGenerationMixin", "enable_ptd"]
