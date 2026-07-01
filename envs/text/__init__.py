"""The text / discrete-LM env family — two tokenizer scales of the SAME diffusion-LM type.

Importing this package registers BOTH environments (their heavy deps are imported lazily, so this
is cheap and needs no extras):
  - ``char_text``   (``envs.text.char``)        — char-level, tiny built-in corpus, dependency-free.
  - ``tinystories`` (``envs.text.tinystories``) — real GPT-2 BPE over streamed TinyStories (the
    ``text`` uv extra; deps imported lazily at use, not at import).

Both run the identical rig — absorbing schedule + transformer + ``d3pm``/``mdlm``/``sedd`` — and
differ only in tokenizer, vocabulary, corpus, and scale. Loading the ``envs.text`` plugin makes both
available; an experiment then selects one via ``environment: {name: char_text | tinystories}``.
"""

from . import char, tinystories  # noqa: F401  — fire the @register decorators for both envs

__all__ = ["char", "tinystories"]
