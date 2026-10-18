# Transformers integration status

Direct integration into the Transformers library is paused. PTD's primary
entry point remains the experimental, instance-local `model.ptd_generate()`
API documented in the README. The model tokenizer must be passed explicitly
to `tree_draft.enable_ptd(model, tokenizer=tokenizer)`.

The older `model.generate(..., use_ptd=True)` compatibility wrapper and the
Transformers `v4.54.0` patch in `patches/` are retained as research artifacts;
they are not part of the currently supported workflow. In particular, do not
assume full parity with Transformers generation features. Custom logits
processors, streamers, and structured generation outputs are not supported by
the current PTD search loop.

Before revisiting native integration, the PTD API needs model-level regression
tests for token correctness, cache updates, stopping behavior, and supported
generation options on the target architectures.
