from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar

from anytrain._compat import Self

from ._deps import training_classes

if TYPE_CHECKING:
    from tokenizers import Tokenizer

PRIVATE_USE_RANGES = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)

Merge = tuple[int, int, int]
_ProgressT = TypeVar("_ProgressT")
STATIC_PROGRESS_INTERVAL_SECONDS = 60.0


class CoreBPE:
    def __init__(
        self,
        tokens: Mapping[int, Sequence[int]],
        merges: Sequence[Merge] = (),
    ) -> None:
        if not tokens:
            raise ValueError("tokens must not be empty")

        self.tokens = {
            int(token_id): tuple(base_ids)
            for token_id, base_ids in tokens.items()
        }
        for token_id, base_ids in self.tokens.items():
            if not base_ids:
                raise ValueError(f"token {token_id} must contain at least one frame")

        self.merges = tuple(merges)
        self.base_to_id = self.build_base_to_id(self.tokens)

    @property
    def vocab_size(self) -> int:
        # Assumes token ids are contiguous from 0, which holds for ids produced
        # by tokenizers training. Callers that inject sparse ids must remap them
        # before relying on this bound.
        return max(self.tokens) + 1

    @classmethod
    def train(
        cls,
        corpus: Callable[[], Iterable[Sequence[int]]],
        *,
        base: Iterable[int] | None = None,
        vocab_size: int = 30_000,
        min_frequency: int = 0,
        show_progress: bool = True,
        max_token_length: int | None = None,
    ) -> Self:
        if base is None:
            base_ids, num_training_sequences = scan_base_corpus(
                corpus,
                show_progress=show_progress,
            )
        else:
            base_ids = set(base)
            num_training_sequences = None
            write_progress(
                "CodecBPE alphabet: skipped for single codebook",
                enabled=show_progress,
            )
        base_tokens = private_use_tokens(base_ids)
        tokenizer = train_tokenizers_bpe(
            text_corpus(corpus, base_tokens),
            base_tokens=base_tokens,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            max_token_length=max_token_length,
            length=num_training_sequences,
        )
        tokens, merges = core_state_from_tokenizer(tokenizer, base_tokens)
        return cls(tokens, merges)

    def decode(self, token_ids: Sequence[int]) -> list[int]:
        if not token_ids:
            raise ValueError("token_ids must not be empty")

        base_ids: list[int] = []
        for token_id in token_ids:
            base_ids.extend(self.tokens[token_id])
        return base_ids

    def token_lengths(self) -> dict[int, int]:
        return {token_id: len(base_ids) for token_id, base_ids in self.tokens.items()}

    @staticmethod
    def build_base_to_id(tokens: Mapping[int, tuple[int, ...]]) -> dict[int, int]:
        base_to_id: dict[int, int] = {}
        for token_id, base_ids in tokens.items():
            if len(base_ids) == 1:
                base_to_id[base_ids[0]] = token_id
        return base_to_id


def private_use_char(index: int) -> str:
    for start, end in PRIVATE_USE_RANGES:
        size = end - start + 1
        if index < size:
            return chr(start + index)
        index -= size
    raise ValueError("private-use character index out of range")


def private_use_capacity() -> int:
    return sum(end - start + 1 for start, end in PRIVATE_USE_RANGES)


def private_use_tokens(base_ids: Iterable[int]) -> dict[int, str]:
    return {
        base_id: private_use_char(index)
        for index, base_id in enumerate(sorted(base_ids))
    }


def dynamic_progress(enabled: bool) -> bool:
    if not enabled:
        return False
    isatty = getattr(sys.stderr, "isatty", None)
    return bool(isatty is not None and isatty())


def write_progress(message: str, *, enabled: bool) -> None:
    if not enabled:
        return

    if dynamic_progress(enabled):
        from tqdm.auto import tqdm

        tqdm.write(message)
    else:
        print(message, file=sys.stderr, flush=True)


class ProgressLog:
    """Write periodic static progress without emitting terminal control output."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled and not dynamic_progress(enabled)
        self.last_write = time.monotonic()

    def write(self, message: str) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_write < STATIC_PROGRESS_INTERVAL_SECONDS:
            return
        self.last_write = now
        write_progress(message, enabled=True)


def progress(
    iterable: Iterable[_ProgressT],
    *,
    enabled: bool,
    desc: str,
) -> Iterable[_ProgressT]:
    if not enabled:
        return iterable
    if dynamic_progress(enabled):
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc)

    progress_log = ProgressLog(enabled=True)

    def logged() -> Iterable[_ProgressT]:
        for count, item in enumerate(iterable, start=1):
            yield item
            progress_log.write(f"{desc}: {count:,} sequences processed")

    return logged()


def scan_base_corpus(
    corpus: Callable[[], Iterable[Sequence[int]]],
    *,
    show_progress: bool,
) -> tuple[set[int], int]:
    base: set[int] = set()
    num_sequences = 0
    for seq in progress(corpus(), enabled=show_progress, desc="CodecBPE alphabet"):
        ids = tuple(seq)
        num_sequences += 1
        if not ids:
            raise ValueError("corpus must not contain empty sequences")
        base.update(ids)

    if num_sequences == 0:
        raise ValueError("corpus must not be empty")
    if not base:
        raise ValueError("corpus must contain at least one frame")
    return base, num_sequences


def text_corpus(
    corpus: Callable[[], Iterable[Sequence[int]]],
    base_tokens: Mapping[int, str],
) -> Iterable[str]:
    num_sequences = 0
    for seq in corpus():
        ids = tuple(seq)
        num_sequences += 1
        if not ids:
            raise ValueError("corpus must not contain empty sequences")
        yield base_text(ids, base_tokens)
    if num_sequences == 0:
        raise ValueError("corpus must not be empty")


def base_text(base_ids: Sequence[int], base_tokens: Mapping[int, str]) -> str:
    pieces: list[str] = []
    for base_id in base_ids:
        if base_id not in base_tokens:
            raise KeyError(f"unknown frame id: {base_id}")
        pieces.append(base_tokens[base_id])
    return "".join(pieces)


def train_tokenizers_bpe(
    corpus: Iterable[str],
    *,
    base_tokens: Mapping[int, str],
    vocab_size: int,
    min_frequency: int,
    show_progress: bool,
    max_token_length: int | None,
    length: int | None,
) -> Tokenizer:
    Tokenizer, BPE, BpeTrainer = training_classes()
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=dynamic_progress(show_progress),
        max_token_length=max_token_length,
        initial_alphabet=list(base_tokens.values()),
    )
    write_progress(
        "CodecBPE trainer: started (corpus, pair counts, merges)",
        enabled=show_progress,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer, length=length)
    write_progress("CodecBPE trainer: completed", enabled=show_progress)
    return tokenizer


def core_state_from_tokenizer(
    tokenizer: Tokenizer,
    base_tokens: Mapping[int, str],
) -> tuple[dict[int, tuple[int, ...]], list[Merge]]:
    model = json.loads(tokenizer.to_str())["model"]
    char_to_base = {text: base_id for base_id, text in base_tokens.items()}
    vocab = {text: int(token_id) for text, token_id in model["vocab"].items()}
    tokens = {
        token_id: tuple(char_to_base[char] for char in text) for text, token_id in vocab.items()
    }
    merges: list[Merge] = []
    for left_text, right_text in model["merges"]:
        token_text = left_text + right_text
        merges.append((vocab[left_text], vocab[right_text], vocab[token_text]))
    return tokens, merges
