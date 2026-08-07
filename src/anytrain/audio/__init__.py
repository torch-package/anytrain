"""Audio I/O adapters and frame-sequence BPE.

Import from explicit submodules:

- ``anytrain.audio.codec``: waveform codec backends, registry, and protocols
- ``anytrain.audio.tokenizer`` / ``anytrain.audio.detokenizer``: direction views
- ``anytrain.audio.bpe``: int BPE over discrete codec frame sequences

``AudioCodec`` is the bidirectional composition of tokenizer and detokenizer
views over a shared backend. BPE operates on codes only and does not load a
neural codec.
"""

__all__: list[str] = []
