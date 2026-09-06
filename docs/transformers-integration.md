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

## Planned fork integration

The next integration step is a small change to a pinned Transformers checkout
(currently `4.54.0`): add a PTD generation mode to `GenerationMixin.generate`
and route it to the model's `_ptd_search` hook. The PTD algorithm and model
backends stay in this repository, so the fork remains a thin dispatch layer.
The instance-local API above is kept stable so code can move between the
compatibility layer and the fork without changing call sites.

The fork should preserve the normal generation contract, including logits
processors, stopping criteria, streamers, `return_dict_in_generate`, and
`past_key_values`. Architecture-specific attention-mask and cache handling
belongs in PTD backends rather than in the generic Transformers dispatcher.
