# Transformers integration

PTD exposes a Transformers-style, opt-in generation path without monkey
patching the global `GenerationMixin`:

```python
import tree_draft
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
tree_draft.enable_ptd(model)

inputs = tokenizer("Explain speculative decoding in one paragraph.", return_tensors="pt")
outputs = model.generate(
    **inputs,
    use_ptd=True,
    max_new_tokens=128,
    temperature=0.0,
    ptd_config=tree_draft.PTDDecodeConfig(),
)
```

`use_ptd` is consumed by PTD's instance-local dispatcher. Calls without it
are forwarded unchanged to the model's original Transformers `generate`
method. `ptd_config` accepts either `PTDDecodeConfig` or a mapping compatible
with `configs/tree_config.json`.

## Transformers fork

The companion `transformers-ptd` checkout is based on Transformers `v4.54.0`,
branch `codex/ptd-integration`. It adds the
`use_ptd=True` dispatch to `GenerationMixin.generate` and routes it to the
model's `_ptd_search` hook. The PTD algorithm and model backends stay in this
repository, so the fork remains a thin dispatch layer.

The reproducible patch is checked in at
`patches/transformers-v4.54.0-ptd-dispatch.patch`. Apply it to a clean
Transformers `v4.54.0` checkout to recreate the fork branch:

```bash
git checkout v4.54.0
git switch -c codex/ptd-integration
git apply /path/to/PTD/patches/transformers-v4.54.0-ptd-dispatch.patch
```

Install both projects in editable mode, then use the same call shown above:

```bash
pip install -e /path/to/transformers-ptd
pip install -e "/path/to/PTD[runtime]"
```

The instance-local API remains compatible with standard Transformers, so code
can move between the compatibility layer and the fork without changing call
sites.

The fork preserves the normal generation contract, including logits
processors, stopping criteria, streamers, `return_dict_in_generate`, and
`past_key_values`. Architecture-specific attention-mask and cache handling
belongs in PTD backends rather than in the generic Transformers dispatcher.
