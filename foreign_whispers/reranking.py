"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


# ---------------------------------------------------------------------------
# Duration-aware re-ranking helpers
# ---------------------------------------------------------------------------

# Assumed TTS speaking rate for Spanish (characters per second).
# Matches the heuristic used in alignment._estimate_duration.
_CHARS_PER_SECOND = 15.0

# Spanish verbose phrases and their shorter equivalents.
# Applied first so filler removal operates on already-contracted text.
# Ordered longest-first to avoid partial substring matches.
_CONTRACTIONS: dict[str, str] = {
    "en lo que respecta a":    "sobre",
    "con la finalidad de":     "para",
    "poner de manifiesto":     "mostrar",
    "con el fin de":           "para",
    "en el caso de que":       "si",
    "en la actualidad":        "hoy",
    "a pesar de que":          "aunque",
    "con respecto a":          "sobre",
    "en relación con":         "sobre",
    "hacer referencia":        "referir",
    "llevar a cabo":           "hacer",
    "tener en cuenta":         "considerar",
    "dar cuenta de":           "explicar",
    "en este momento":         "ahora",
    "por medio de":            "via",
    "a causa de":              "por",
    "debido a":                "por",
    "se encuentra":            "está",
    "se encuentran":           "están",
}

# Spanish filler/hedge words that can be dropped without losing meaning.
# Ordered longest-first so multi-word fillers are matched before single words.
_FILLER_PHRASES: list[str] = [
    "por supuesto",   "sin embargo",    "no obstante",    "por otro lado",
    "a pesar de ello","en cualquier caso","al fin y al cabo","en definitiva",
    "en resumen",     "de todas formas","por lo tanto",   "en realidad",
    "de hecho",       "es decir",       "o sea",          "así que",
    "además",         "también",        "incluso",        "simplemente",
    "básicamente",    "claramente",     "obviamente",     "realmente",
    "actualmente",    "generalmente",   "normalmente",
]

# English filler/hedge words stripped before re-translation.
_EN_FILLERS = (
    r"basically|actually|really|simply|just|also|even|"
    r"clearly|obviously|generally|normally|currently"
)


def _apply_contractions(text: str) -> str:
    """Replace verbose Spanish phrases with shorter equivalents."""
    result = text
    for long_form, short_form in _CONTRACTIONS.items():
        result = re.sub(
            re.escape(long_form), short_form, result, flags=re.IGNORECASE
        )
    return result


def _remove_fillers(text: str) -> str:
    """Strip Spanish filler/hedge phrases that add length without meaning."""
    result = text
    for filler in _FILLER_PHRASES:
        # Match whole phrase, optionally followed by a comma and whitespace
        result = re.sub(
            r'\b' + re.escape(filler) + r'\b[,]?\s*',
            ' ',
            result,
            flags=re.IGNORECASE,
        )
    # Normalise whitespace and remove any leading punctuation left behind
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r'^[,;]\s*', '', result)
    return result


def _truncate_to_budget(text: str, budget_chars: int) -> str:
    """Hard-truncate *text* to *budget_chars*, breaking at a word boundary."""
    if len(text) <= budget_chars:
        return text
    truncated = text[:budget_chars]
    last_space = truncated.rfind(' ')
    # Only break at word boundary if it leaves at least half the budget
    if last_space > budget_chars // 2:
        truncated = truncated[:last_space]
    # Strip trailing punctuation fragments
    return truncated.rstrip('.,;:!?¿¡ ')


def _retranslate(source_text: str) -> str | None:
    """Re-translate *source_text* EN→ES after stripping English fillers.

    Returns the new translation, or ``None`` if argostranslate is unavailable
    or the simplified source is identical to the original.
    """
    # Strip English fillers to produce a shorter source sentence
    short_source = re.sub(
        r'\b(' + _EN_FILLERS + r')\b[,]?\s*',
        ' ',
        source_text,
        flags=re.IGNORECASE,
    )
    short_source = re.sub(r'\s+', ' ', short_source).strip()
    if short_source == source_text:
        # Nothing was removed — re-translating would give the same result
        return None

    try:
        from argostranslate import translate  # type: ignore
        installed = translate.get_installed_languages()
        en = next((lang for lang in installed if lang.code == 'en'), None)
        es = next((lang for lang in installed if lang.code == 'es'), None)
        if en is None or es is None:
            return None
        translation = en.get_translation(es)
        if translation is None:
            return None
        return translation.translate(short_source)
    except Exception as exc:
        logger.debug("Re-translation failed: %s", exc)
        return None


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Implementation strategy (hybrid rule-based):

    1. **Contractions** — replace verbose Spanish phrases with shorter forms
       (e.g. "en este momento" → "ahora", "llevar a cabo" → "hacer").
    2. **Filler removal** — strip hedge words that add length without meaning
       (e.g. "básicamente", "en realidad", "sin embargo").
    3. **Combined** — apply both contractions and filler removal together.
    4. **Re-translation** — strip English filler words from the source and
       re-translate; argostranslate often produces a shorter output.
    5. **Hard truncation** — as a last resort, truncate to the character
       budget at a word boundary.

    Candidates are deduplicated and sorted shortest-first.

    Returns:
        List of ``TranslationCandidate`` items, shortest first.
        Empty list only if no candidate differs from the baseline.
    """
    budget_chars = int(target_duration_s * _CHARS_PER_SECOND)
    candidates: list[TranslationCandidate] = []
    seen: set[str] = {baseline_es}  # deduplicate against baseline and each other

    def _add(text: str, rationale: str) -> None:
        """Add a candidate if it is non-empty and not already seen."""
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            candidates.append(TranslationCandidate(
                text=text,
                char_count=len(text),
                brevity_rationale=rationale,
            ))

    # --- Step 1: contraction substitution ---
    contracted = _apply_contractions(baseline_es)
    if contracted != baseline_es:
        _add(contracted, "replaced verbose phrases with shorter contractions")

    # --- Step 2: filler removal ---
    no_fillers = _remove_fillers(baseline_es)
    if no_fillers != baseline_es:
        _add(no_fillers, "removed filler/hedge phrases")

    # --- Step 3: contractions + filler removal combined ---
    combined = _remove_fillers(_apply_contractions(baseline_es))
    _add(combined, "contracted phrases and removed fillers")

    # --- Step 4: re-translate from simplified English source ---
    alt = _retranslate(source_text)
    if alt:
        _add(alt, "re-translated from filler-stripped English source")
        # Also contract the alternative translation
        alt_contracted = _apply_contractions(alt)
        _add(alt_contracted, "re-translated and contracted")

    # --- Step 5: hard truncation to character budget (last resort) ---
    truncated = _truncate_to_budget(baseline_es, budget_chars)
    if truncated:
        _add(
            truncated,
            f"truncated to {budget_chars}-char budget "
            f"({target_duration_s:.1f}s × {_CHARS_PER_SECOND:.0f} chars/s)",
        )

    # Sort shortest-first so the caller can easily pick the best fit
    candidates.sort(key=lambda c: c.char_count)

    logger.info(
        "get_shorter_translations: budget=%.1fs (%d chars), "
        "baseline=%d chars, produced %d candidates",
        target_duration_s,
        budget_chars,
        len(baseline_es),
        len(candidates),
    )
    return candidates