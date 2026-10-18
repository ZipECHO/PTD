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


def test_ptd_config_rejects_invalid_values():
    import pytest
    from tree_draft import PTDDecodeConfig
    from tree_draft.generation import _coerce_ptd_config

    with pytest.raises(ValueError, match="max_depth"):
        PTDDecodeConfig(max_depth=0)
    with pytest.raises(ValueError, match="tree_update_method"):
        PTDDecodeConfig(tree_update_method="unknown")
    with pytest.raises(ValueError, match="Unknown PTD config field"):
        _coerce_ptd_config({"max_dept": 6})


def test_ptd_generate_reports_tokenizer_and_input_contract_early():
    import pytest
    import tree_draft

    class TensorLike:
        shape = (1, 2)

    class DummyModel:
        generation_config = type("GenerationConfig", (), {"temperature": 0.0, "do_sample": False})()

    model = tree_draft.enable_ptd(DummyModel())
    with pytest.raises(RuntimeError, match="pass it to enable_ptd"):
        model.ptd_generate(input_ids=TensorLike(), max_new_tokens=1)

    model = tree_draft.enable_ptd(DummyModel(), tokenizer=object())
    with pytest.raises(ValueError, match="rank-2"):
        model.ptd_generate(input_ids=[1, 2], max_new_tokens=1)
    with pytest.raises(ValueError, match="positive integer"):
        model.ptd_generate(input_ids=TensorLike(), max_new_tokens=0)


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
