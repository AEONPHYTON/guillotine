"""
response_parser.py

Extracts the final word from a model's raw response, handling both clean
responses ("CHAIR") and noisy ones (explanations, reasoning traces between
<think>...</think>, quotes, punctuation).
"""
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
UPPERCASE_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
QUOTED_WORD_PATTERN = re.compile(r"[\"']([A-Za-z]+)[\"']")
WORD_STRIP_CHARS = " \t\n\r.,!?;:'\"()[]{}*_"
TRAILING_STOPWORDS = {"because", "since", "as", "so", "therefore", "thus", "hence", "and", "but"}


@dataclass
class ParsedResponse:
    predicted_word: str
    format_compliant: bool


def strip_thinking(text: str) -> str:
    """Removes any leftover <think>...</think> blocks from the raw text.

    Needed even though the Ollama version in use already separates
    thinking at the API level (a distinct "thinking" field from
    "response"): on some model/version combinations the tag still ends up
    in the response text. On already-clean responses this is a no-op.
    """
    return THINK_BLOCK_PATTERN.sub("", text).strip()


def extract_predicted_word(raw_text: str) -> ParsedResponse:
    """
    Extracts the final word from a raw response.

    format_compliant=True if, after removing the reasoning, the cleaned
    text was already a single word (no heuristic extraction needed) -
    False if extraction from longer text was required.
    """
    cleaned = strip_thinking(raw_text)

    if not cleaned:
        return ParsedResponse(predicted_word="", format_compliant=False)

    tokens = cleaned.split()
    if len(tokens) == 1:
        word = tokens[0].strip(WORD_STRIP_CHARS)
        return ParsedResponse(predicted_word=word, format_compliant=True)

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ParsedResponse(predicted_word="", format_compliant=False)

    # Common pattern "ANSWER\n\nExplanation: ..." - the model answers
    # correctly on the first line and then adds text it shouldn't. If the
    # first line is a single token, it's almost certainly the real
    # answer - much more reliable than searching the last line in this
    # case (which would grab the last word of the explanation).
    first_line_tokens = lines[0].split()
    if len(first_line_tokens) == 1:
        word = first_line_tokens[0].strip(WORD_STRIP_CHARS)
        return ParsedResponse(predicted_word=word, format_compliant=False)

    # Opposite pattern "reasoning then ANSWER", or an answer in quotes/
    # uppercase somewhere in the middle of the text: search the last line.
    last_line = lines[-1]

    quoted_matches = QUOTED_WORD_PATTERN.findall(last_line)
    if quoted_matches:
        return ParsedResponse(predicted_word=quoted_matches[-1], format_compliant=False)

    uppercase_matches = UPPERCASE_WORD_PATTERN.findall(last_line)
    if uppercase_matches:
        return ParsedResponse(predicted_word=uppercase_matches[-1], format_compliant=False)

    # Last resort before falling back to the last token: uppercase words
    # anywhere in the cleaned text, not just the last line (covers
    # responses with the word isolated elsewhere in the text, e.g.
    # halfway through).
    uppercase_anywhere = UPPERCASE_WORD_PATTERN.findall(cleaned)
    if uppercase_anywhere:
        return ParsedResponse(predicted_word=uppercase_anywhere[0], format_compliant=False)

    # Last resort: the last token of the line, skipping trailing
    # conjunctions like "..., because ..." which would otherwise be
    # picked up by mistake.
    fallback_tokens = [t.strip(WORD_STRIP_CHARS) for t in last_line.split()]
    fallback_tokens = [t for t in fallback_tokens if t]
    while fallback_tokens and fallback_tokens[-1].lower() in TRAILING_STOPWORDS:
        fallback_tokens.pop()
    if fallback_tokens:
        return ParsedResponse(predicted_word=fallback_tokens[-1], format_compliant=False)

    logger.warning(f"Could not extract a word from: {raw_text[:200]!r}")
    return ParsedResponse(predicted_word="", format_compliant=False)
