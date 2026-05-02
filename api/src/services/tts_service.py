"""HTTP-agnostic service wrapping TTS engine functions."""

import pathlib
from pathlib import Path
from typing import Any

from api.src.services.tts_engine import text_file_to_speech as tts_text_file_to_speech


class TTSService:
    """Thin wrapper around the TTS pipeline."""

    def __init__(self, ui_dir: Path, tts_engine: Any) -> None:
        self.ui_dir = ui_dir
        self.tts_engine = tts_engine

    def build_speaker_voice_map(self, speakers: list[str], lang: str) -> dict[str, str]:
        """Map speaker labels to reference WAV paths using round-robin assignment.

        Looks for WAV files in pipeline_data/speakers/{lang}/.
        Returns empty dict if no WAVs are found (falls back to default voice).
        """
        speakers_dir = self.ui_dir / "speakers" / lang
        if not speakers_dir.exists():
            return {}
        wavs = sorted(speakers_dir.glob("*.wav"))
        if not wavs:
            return {}
        return {
            speaker: str(wavs[i % len(wavs)])
            for i, speaker in enumerate(sorted(speakers))
        }

    def text_file_to_speech(
        self,
        source_path: str,
        output_path: str,
        *,
        alignment: bool | None = None,
        speaker_voice_map: dict[str, str] | None = None,
    ) -> None:
        """Generate time-aligned TTS audio from a translated JSON transcript."""
        tts_text_file_to_speech(
            source_path,
            output_path,
            self.tts_engine,
            alignment=alignment,
            speaker_voice_map=speaker_voice_map,
        )

    @staticmethod
    def title_for_video_id(video_id: str, search_dir: pathlib.Path) -> str | None:
        for f in search_dir.glob("*.json"):
            return f.stem
        return None

    def compute_alignment(
        self,
        en_transcript: dict,
        es_transcript: dict,
        silence_regions: list[dict],
        max_stretch: float = 1.4,
    ) -> list:
        from foreign_whispers.alignment import compute_segment_metrics, global_align
        metrics = compute_segment_metrics(en_transcript, es_transcript)
        return global_align(metrics, silence_regions, max_stretch)