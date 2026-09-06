import subprocess
import sys

from main import build_parser


def test_model_path_is_not_machine_specific():
    args = build_parser().parse_args([])
    assert args.model_path is None


def test_help_does_not_require_runtime_dependencies():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--model-path" in result.stdout


def test_generation_interface_is_lazy():
    import tree_draft

    assert callable(tree_draft.enable_ptd)
    config = tree_draft.PTDDecodeConfig(max_depth=3)
    assert config.max_depth == 3
    assert config.to_tree_config()["node_config"]["default_max_child_num"] == 4


def test_enable_ptd_is_instance_local():
    import tree_draft

    class DummyModel:
        pass

    model = tree_draft.enable_ptd(DummyModel())
    assert callable(model.ptd_generate)
    assert callable(model._ptd_search)
    assert not hasattr(DummyModel, "ptd_generate")


def test_enable_ptd_adds_opt_in_generate_dispatch():
    import tree_draft

    class DummyModel:
        def generate(self, *args, **kwargs):
            return "standard", args, kwargs

    model = tree_draft.enable_ptd(DummyModel())
    model.ptd_generate = lambda **kwargs: ("ptd", kwargs)

    assert model.generate("tokens", do_sample=False) == (
        "standard", ("tokens",), {"do_sample": False}
    )
    assert model.generate(
        input_ids="tokens", use_ptd=True, ptd_config={"max_depth": 3}
    ) == (
        "ptd", {"input_ids": "tokens", "ptd_config": {"max_depth": 3}}
    )


def test_enable_ptd_preserves_native_transformers_dispatch():
    import tree_draft

    class DummyModel:
        def generate(self, inputs=None, use_ptd=False, **kwargs):
            return inputs, use_ptd, kwargs

    model = tree_draft.enable_ptd(DummyModel())
    assert model.generate("tokens", use_ptd=True) == ("tokens", True, {})
    assert not hasattr(model, "_ptd_generate_dispatch_installed")


def test_cache_server_argument_validation():
    import pytest
    from tree_draft.cache import _launch_cache_server, _repo_cache_server_script

    assert _repo_cache_server_script().endswith("cache_server.py")
    with pytest.raises(ValueError, match="batch_size"):
        _launch_cache_server(0, None, None, timeout=0.1, auto_cleanup=True)
    with pytest.raises(ValueError, match="provided together"):
        _launch_cache_server(1, "corpus", None, timeout=0.1, auto_cleanup=True)
