"""Encode preprocessed EveryVoice token sequences as StyleTTS2 embedding indices.

Text arriving here has already been normalized and (if applicable) converted to
IPA phones by ``everyvoice preprocess``.  The only job of this module is the
index translation:

    EveryVoice escaped token string  →  list[int] (StyleTTS2 embedding indices)

No cleaners, no G2P, no TextProcessor pipeline at encode time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from everyvoice.text.features import DEFAULT_PUNCTUATION_HASH

if TYPE_CHECKING:
    from everyvoice.config.text_config import TextConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EveryVoice internal punctuation token → nearest StyleTTS2 character
# ---------------------------------------------------------------------------
# StyleTTS2's _punctuation = ';:,.!?¡¿—…"«»"" '
# Keys are sourced from DEFAULT_PUNCTUATION_HASH so a KeyError is raised at
# import time if EveryVoice ever renames a token.
# Tokens not representable in StyleTTS2 (e.g. <PAREN>) map to None and are
# dropped with a warning.

_EV_PUNCT_TO_ST2: dict[str, str | None] = {
    DEFAULT_PUNCTUATION_HASH["exclamations"]: "!",
    DEFAULT_PUNCTUATION_HASH["question_symbols"]: "?",
    DEFAULT_PUNCTUATION_HASH["quotemarks"]: "“",  # “ left double quotation mark (U+201C)
    DEFAULT_PUNCTUATION_HASH["periods"]: ".",
    DEFAULT_PUNCTUATION_HASH["commas"]: ",",
    DEFAULT_PUNCTUATION_HASH["colons"]: ":",
    DEFAULT_PUNCTUATION_HASH["semi_colons"]: ";",
    DEFAULT_PUNCTUATION_HASH["hyphens"]: "—",     # — em dash
    DEFAULT_PUNCTUATION_HASH["ellipses"]: "…",    # … horizontal ellipsis
    # Parentheses have no StyleTTS2 equivalent.  Warn and drop.
    DEFAULT_PUNCTUATION_HASH["parentheses"]: None,
}

# Tokens that are silently dropped (None target) — warn users once per encoder.
_WARN_ONCE: set[str] = set()


class EVStyleTTS2TextEncoder:
    """Translate preprocessed EveryVoice token strings to StyleTTS2 indices.

    Parameters
    ----------
    text_config:
        The EveryVoice ``TextConfig``.  Used only to build the set of
        declared symbols for validation — no encoding is done via TextProcessor.
    pretrained_symbols:
        Ordered list of symbols matching the pretrained text-encoder embedding
        table (``StyleTTS2PretrainedConfig.pretrained_symbols``).  Index *i*
        of this list is embedding row *i*.
    """

    def __init__(self, text_config: TextConfig, pretrained_symbols: list[str]) -> None:
        self._pretrained_dict: dict[str, int] = {
            s: i for i, s in enumerate(pretrained_symbols)
        }
        # Merge direct symbol lookup with punctuation remapping.
        self._token_to_idx: dict[str, int] = dict(self._pretrained_dict)
        for ev_token, st2_char in _EV_PUNCT_TO_ST2.items():
            if st2_char is not None and st2_char in self._pretrained_dict:
                self._token_to_idx[ev_token] = self._pretrained_dict[st2_char]
            # Tokens mapping to None (or chars absent from the table) are left
            # out of _token_to_idx so they hit the drop-with-warning path below.

    def encode_token_sequence(self, token_str: str) -> list[int]:
        """Convert an escaped EveryVoice token string to StyleTTS2 indices.

        Parameters
        ----------
        token_str:
            A ``/``-separated sequence of tokens as stored in the
            ``character_tokens`` or ``phone_tokens`` filelist column,
            e.g. ``"h/ɛ/l/oʊ/<COMMA>/w/ɝ/l/d/<EXCL>"``.

        Returns
        -------
        list[int]
            StyleTTS2-compatible embedding indices, one per token, with
            unresolvable tokens dropped (and a warning emitted once per
            unique dropped token).
        """
        indices: list[int] = []
        for token in token_str.split("/"):
            if not token:
                continue
            idx = self._token_to_idx.get(token)
            if idx is None:
                if token not in _WARN_ONCE:
                    _WARN_ONCE.add(token)
                    # Give actionable advice for the two known problematic tokens.
                    extra = ""
                    if token == "<PAREN>":
                        extra = (
                            " Parentheses have no equivalent in the pretrained StyleTTS2 "
                            "symbol table. Add a 'to_replace' rule in your TextConfig to "
                            "replace parenthesis characters before preprocessing."
                        )
                    elif token == "<EPS>":
                        extra = (
                            " Ellipsis has no equivalent in the pretrained StyleTTS2 "
                            "symbol table. Add a 'to_replace' rule in your TextConfig to "
                            "replace ellipsis characters before preprocessing."
                        )
                    logger.warning(
                        "Token %r has no mapping in the pretrained StyleTTS2 symbol "
                        "table and will be silently dropped.%s",
                        token,
                        extra,
                    )
                continue
            indices.append(idx)
        return indices
