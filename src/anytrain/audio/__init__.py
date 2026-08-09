"""Audio I/O adapters and frame-sequence BPE.

Import from explicit submodules:

- ``anytrain.audio.codec``: waveform backend presets and registry
- ``anytrain.audio.tokenizer`` / ``anytrain.audio.detokenizer``: directional ABCs
- ``anytrain.audio.bpe``: int BPE over discrete codec frame sequences

Default preset tokenizer and detokenizer adapters may share a cached backend
instance internally; callers depend only on their directional contracts.
BPE operates on codes only and does not load a neural codec.
"""

__all__: list[str] = []
