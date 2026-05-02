"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }

def dubbing_scorecard(
    metrics:      list[SegmentMetrics],
    aligned:      list[AlignedSegment],
    align_report: dict,
) -> dict:
    """Score a dubbed clip across four quality dimensions, each in [0, 1].

    Higher is always better.  All scores are derived from data already
    computed by ``compute_segment_metrics`` and ``clip_evaluation_report``
    — no external model calls required.

    Dimensions
    ----------
    timing_accuracy
        How well TTS durations fit their source windows.
        Derived from ``mean_abs_duration_error_s`` and ``pct_severe_stretch``.
        Perfect score = 0s error, 0% severe stretch.

    intelligibility_proxy
        Proxy for speech clarity based on speaking rate consistency.
        Segments with extreme stretch (>1.3x) are harder to understand.
        Score = fraction of segments with stretch_factor <= 1.3.

    semantic_fidelity_proxy
        Proxy for meaning preservation based on character-count ratio.
        A translation that is <50% or >150% the length of the source
        likely lost or added significant meaning.
        Score = fraction of segments within the 50–150% length ratio band.

    naturalness
        Speaking rate variance across segments — consistent rate sounds
        natural; wild swings between fast and slow sound robotic.
        Score derived from coefficient of variation of predicted_tts_s /
        source_duration_s ratios (lower variance = higher score).

    Returns
    -------
    dict with keys: timing_accuracy, intelligibility_proxy,
    semantic_fidelity_proxy, naturalness, overall.
    ``overall`` is the unweighted mean of the four dimension scores.
    """
    if not metrics or not aligned:
        return {
            "timing_accuracy":        0.0,
            "intelligibility_proxy":  0.0,
            "semantic_fidelity_proxy": 0.0,
            "naturalness":            0.0,
            "overall":                0.0,
        }

    n = len(metrics)

    # ------------------------------------------------------------------
    # 1. Timing accuracy
    #    Penalise mean duration error (capped at 5s = score 0) and
    #    severe stretch fraction.
    # ------------------------------------------------------------------
    mean_err   = align_report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = align_report.get("pct_severe_stretch", 0.0)
    timing_err_score     = max(0.0, 1.0 - mean_err / 5.0)
    timing_severe_score  = max(0.0, 1.0 - pct_severe / 100.0)
    timing_accuracy      = round((timing_err_score + timing_severe_score) / 2, 3)

    # ------------------------------------------------------------------
    # 2. Intelligibility proxy
    #    Segments stretched beyond 1.3x become noticeably faster and
    #    harder to follow.
    # ------------------------------------------------------------------
    n_intelligible       = sum(1 for a in aligned if a.stretch_factor <= 1.3)
    intelligibility_proxy = round(n_intelligible / n, 3)

    # ------------------------------------------------------------------
    # 3. Semantic fidelity proxy
    #    Target/source character-count ratio should stay in [0.5, 1.5].
    #    Outside that band the translation is suspiciously short or bloated.
    # ------------------------------------------------------------------
    def _ratio_ok(m: SegmentMetrics) -> bool:
        if m.src_char_count == 0:
            return True
        ratio = m.tgt_char_count / m.src_char_count
        return 0.5 <= ratio <= 1.5

    n_faithful            = sum(1 for m in metrics if _ratio_ok(m))
    semantic_fidelity_proxy = round(n_faithful / n, 3)

    # ------------------------------------------------------------------
    # 4. Naturalness
    #    Low variance in stretch factor = consistent speaking rate = natural.
    #    Use coefficient of variation (std/mean) of stretch factors,
    #    capped at 1.0 (= score 0).
    # ------------------------------------------------------------------
    stretches = [a.stretch_factor for a in aligned]
    mean_sf   = _stats.mean(stretches)
    if mean_sf == 0:
        naturalness = 0.0
    else:
        stdev_sf    = _stats.pstdev(stretches)   # population stdev
        cv          = stdev_sf / mean_sf          # coefficient of variation
        naturalness = round(max(0.0, 1.0 - cv), 3)

    # ------------------------------------------------------------------
    # Overall: unweighted mean
    # ------------------------------------------------------------------
    overall = round(
        (timing_accuracy + intelligibility_proxy +
         semantic_fidelity_proxy + naturalness) / 4,
        3,
    )

    return {
        "timing_accuracy":         timing_accuracy,
        "intelligibility_proxy":   intelligibility_proxy,
        "semantic_fidelity_proxy": semantic_fidelity_proxy,
        "naturalness":             naturalness,
        "overall":                 overall,
    }