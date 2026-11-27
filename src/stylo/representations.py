"""Text representations (docs/phase_a_plan.md section 2) and windowing.

Windowing is done on the RAW token stream (so a 500-word window covers the
same stretch of text in every representation); the representation is then
applied token by token. POS tags are computed once per unit, on the whole
unit, and sliced, so the tagger always sees full context.

Codes:
  raw           (a) unmasked control
  fw_pos        (b) function words + punctuation kept, other words -> POS tag
  pos_only      (c) every word -> POS tag, punctuation kept
  fw_only       (d) function words only (lowercased); everything else dropped.
                    Used as the function-word frequency vector by Delta/linear.
  content_only  (e) content words kept, function words -> "X"
  fw_x          sensitivity variant of (b): other words -> "X"
                    (same rule as legacy/cluster/mask_corpus.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Callable, Sequence

from .text import is_word, load_function_words, tokenize

PLACEHOLDER = "X"
REPRESENTATIONS = ("raw", "fw_pos", "pos_only", "fw_only", "content_only", "fw_x")
NEEDS_POS = {"fw_pos", "pos_only"}

_FW = load_function_words()


@dataclass
class Window:
    unit_id: str
    cls: str
    index: int            # window number within the unit
    start_word: int       # raw word offset of the window start
    n_words: int          # raw words covered
    tokens: list[str]     # tokens in the chosen representation
    counts: Counter = field(default_factory=Counter, repr=False)

    def __post_init__(self):
        if not self.counts:
            self.counts = Counter(t.lower() for t in self.tokens if is_word(t))

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


def _is_fw(tok: str) -> bool:
    return tok.lower() in _FW


def apply_representation(rep: str, tokens: Sequence[str],
                         tags: Sequence[str] | None = None) -> list[str]:
    if rep == "raw":
        return list(tokens)
    if rep == "fw_x":
        return [t if (not is_word(t) or _is_fw(t)) else PLACEHOLDER for t in tokens]
    if rep == "content_only":
        return [PLACEHOLDER if (is_word(t) and _is_fw(t)) else t for t in tokens]
    if rep == "fw_only":
        return [t.lower() for t in tokens if is_word(t) and _is_fw(t)]
    if rep in NEEDS_POS:
        if tags is None or len(tags) != len(tokens):
            raise ValueError(f"representation {rep!r} needs one POS tag per token")
        if rep == "fw_pos":
            return [t if (not is_word(t) or _is_fw(t)) else g for t, g in zip(tokens, tags)]
        return [g if is_word(t) else t for t, g in zip(tokens, tags)]
    raise ValueError(f"unknown representation {rep!r}; choose from {REPRESENTATIONS}")


def make_windows(unit_id: str, cls: str, text: str, rep: str, size: int | None,
                 tagger: Callable[[list[str]], list[str]] | None = None) -> list[Window]:
    """Split one unit into non-overlapping windows of `size` raw words
    (None = the whole unit as one window). A trailing remainder shorter than
    half a window is dropped, but every unit keeps at least one window."""
    raw = tokenize(text)
    tags = tagger(raw) if rep in NEEDS_POS else None
    if tags is not None and len(tags) != len(raw):
        raise ValueError("tagger returned a different number of tags than tokens")

    # token index boundaries every `size` words
    bounds = [0]
    words = 0
    for i, t in enumerate(raw):
        if is_word(t):
            if size and words and words % size == 0 and bounds[-1] != i:
                bounds.append(i)
            words += 1
    bounds.append(len(raw))

    spans = list(zip(bounds[:-1], bounds[1:]))
    out: list[Window] = []
    start_word = 0
    for idx, (a, b) in enumerate(spans):
        n = sum(1 for t in raw[a:b] if is_word(t))
        is_last = idx == len(spans) - 1
        if size and is_last and n < size / 2 and out:
            break
        toks = apply_representation(rep, raw[a:b], tags[a:b] if tags is not None else None)
        out.append(Window(unit_id, cls, len(out), start_word, n, toks))
        start_word += n
    return out


def spacy_tagger(model: str = "en_core_web_sm") -> Callable[[list[str]], list[str]]:
    """Universal POS tagger that keeps our tokenization (needs spaCy and the
    model installed). Imported lazily so the rest of the harness needs only
    the standard library."""
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(model, exclude=["parser", "ner", "lemmatizer"])

    def tag(tokens: list[str]) -> list[str]:
        out: list[str] = []
        step = 5000  # keep docs a manageable size
        for i in range(0, len(tokens), step):
            doc = Doc(nlp.vocab, words=list(tokens[i:i + step]))
            for _, proc in nlp.pipeline:
                doc = proc(doc)
            out.extend(t.pos_ for t in doc)
        return out

    tag.model_name = f"spacy {spacy.__version__} / {model}"  # type: ignore[attr-defined]
    return tag
