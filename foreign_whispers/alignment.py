"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.

No external dependencies — stdlib only.
"""
import dataclasses
import re
import unicodedata
from enum import Enum


def _count_syllables(text: str) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    # Normalise: decompose accented chars, keep only ASCII letters + spaces
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiou]+", ascii_text)
    return max(1, len(clusters))


_SYLLABLE_RATE = 4.5  # syllables per second for Romance languages
_PAUSE_COMMA   = 0.12  # seconds per comma
_PAUSE_PERIOD  = 0.18  # seconds per sentence-ending punctuation
_PAUSE_ELLIP   = 0.25  # seconds per ellipsis

def _estimate_duration(text: str) -> float:
    """Estimate TTS duration in seconds using syllable rate + punctuation pauses."""
    syllables = _count_syllables(text)
    speech_s  = syllables / _SYLLABLE_RATE

    # Add pause time for punctuation (TTS engines insert natural pauses)
    pauses  = text.count(",")  * _PAUSE_COMMA
    pauses += len(re.findall(r"[.!?¿¡]", text)) * _PAUSE_PERIOD
    pauses += text.count("…")  * _PAUSE_ELLIP
    pauses += text.count("...") * _PAUSE_ELLIP

    return speech_s + pauses


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text)
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        ))
    return metrics


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - A dynamic-programming or constraint-solver approach would produce
      better schedules, but this is the baseline to start from.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(m.source_end))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)
        # ACCEPT, REQUEST_SHORTER, FAIL → stretch stays at 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned

def global_align_dp(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """DP global alignment minimising total squared stretch penalty.

    Improves on the greedy ``global_align`` by looking ahead: instead of
    making the locally optimal decision for each segment independently, it
    finds the assignment of gap-shifts that minimises the sum of squared
    stretch factors across all segments — subject to the constraint that
    each gap-shift can only consume silence that actually exists after that
    segment.

    With no silence regions available, produces identical output to
    ``global_align``. The advantage appears when real VAD silence data
    is provided: the DP saves gaps for segments that need them most
    rather than consuming them greedily left-to-right.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output dicts with ``{"start_s", "end_s", "label"}``.
            Pass ``[]`` if VAD is unavailable.
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    import math

    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    n = len(metrics)
    if n == 0:
        return []

    # Quantise drift to 0.1s steps; cap state space at 300 steps (30s max drift)
    STEP = 0.1
    max_drift_steps = min(
        int(sum(m.overflow_s for m in metrics if m.overflow_s > 0) / STEP) + 1,
        300,
    )
    INF = math.inf

    # dp[d] = min total squared-stretch cost arriving at this segment with drift = d*STEP
    dp_cur = [INF] * (max_drift_steps + 1)
    dp_cur[0] = 0.0

    # choice[i][d] = True if we took a gap-shift at segment i arriving with drift d
    choice: list[list[bool]] = [
        [False] * (max_drift_steps + 1) for _ in range(n)
    ]

    for i, m in enumerate(metrics):
        dp_next   = [INF] * (max_drift_steps + 1)
        gap_s     = _silence_after(m.source_end)
        gap_steps = int(m.overflow_s / STEP) if m.overflow_s > 0 else 0
        sf        = min(m.predicted_stretch, max_stretch)
        no_gap_cost = max(0.0, sf - 1.0) ** 2  # 0 when segment fits naturally

        for d in range(max_drift_steps + 1):
            if dp_cur[d] == INF:
                continue

            # Option A: no gap-shift — pay stretch cost, drift unchanged
            cost_a = dp_cur[d] + no_gap_cost
            if cost_a < dp_next[d]:
                dp_next[d] = cost_a
                choice[i][d] = False

            # Option B: gap-shift — zero stretch cost, drift increases
            if gap_s >= m.overflow_s and m.overflow_s > 0:
                new_d = d + gap_steps
                if new_d <= max_drift_steps:
                    cost_b = dp_cur[d]   # no stretch penalty
                    if cost_b < dp_next[new_d]:
                        dp_next[new_d] = cost_b
                        choice[i][d] = True

        dp_cur = dp_next

    # Reconstruct decisions by replaying forward through the choice table
    use_gap = [False] * n
    d_state = 0
    for i, m in enumerate(metrics):
        took_gap   = choice[i][d_state]
        use_gap[i] = took_gap
        if took_gap:
            d_state = min(d_state + int(m.overflow_s / STEP), max_drift_steps)

    # Build AlignedSegment list
    aligned:          list[AlignedSegment] = []
    cumulative_drift: float                = 0.0

    for i, m in enumerate(metrics):
        gap_s = _silence_after(m.source_end)

        if use_gap[i] and gap_s >= m.overflow_s and m.overflow_s > 0:
            action    = AlignAction.GAP_SHIFT
            gap_shift = m.overflow_s
            stretch   = 1.0
        else:
            gap_shift = 0.0
            action    = decide_action(m, available_gap_s=0.0)
            stretch   = min(m.predicted_stretch, max_stretch) \
                        if action == AlignAction.MILD_STRETCH else 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = i,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))
        cumulative_drift += gap_shift

    return aligned