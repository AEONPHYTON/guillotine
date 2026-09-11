"""
scoring.py

The three agreed-upon automatic metrics: exact match, same stem (Porter
stemmer), normalized Levenshtein similarity. The two manual metrics
(synonym_manual, coherence_manual) remain empty columns, filled in by hand
after data collection.
"""
import logging
from dataclasses import dataclass

from nltk.stem import PorterStemmer
from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)

_stemmer = PorterStemmer()


@dataclass
class ScoringResult:
    exact_match: int
    same_root: int
    levenshtein_normalized: float
    is_similar_lev: int


def score(predicted_word: str, expected_solution: str, similarity_threshold: float) -> ScoringResult:
    """Computes the three automatic metrics. Case-insensitive comparison."""
    predicted = predicted_word.strip().lower()
    expected = expected_solution.strip().lower()

    if not predicted:
        return ScoringResult(exact_match=0, same_root=0, levenshtein_normalized=1.0, is_similar_lev=0)

    exact_match = int(predicted == expected)

    try:
        same_root = int(_stemmer.stem(predicted) == _stemmer.stem(expected))
    except Exception as e:
        logger.warning(f"Stemming error for '{predicted}'/'{expected}': {e}")
        same_root = 0

    normalized_distance = Levenshtein.normalized_distance(predicted, expected)
    is_similar_lev = int(normalized_distance < similarity_threshold)

    return ScoringResult(
        exact_match=exact_match,
        same_root=same_root,
        levenshtein_normalized=round(normalized_distance, 4),
        is_similar_lev=is_similar_lev,
    )


def is_clue_echo(predicted_word: str, clues: list[str]) -> int:
    """1 if the answer matches (case-insensitive) one of the 5 given clues.

    Flags the case where the model failed to find the hidden connector and
    just echoed back one of the input clues - a different kind of failure
    from an "original" wrong answer (see guillo_09/petrichor: the clue
    'perfume' echoed verbatim, not a synonym found by the model).
    """
    if not predicted_word:
        return 0
    predicted = predicted_word.strip().lower()
    return int(any(predicted == c.strip().lower() for c in clues))
