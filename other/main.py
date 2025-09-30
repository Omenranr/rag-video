#!/usr/bin/env python3
"""
Transcribe and diarize an audio file using faster‑whisper and pyannote, with an
optional speed‑up or slow‑down preprocessing step (FFmpeg `atempo`). Timestamps
in the final SRT are mapped back to the **original** timeline so the subtitles
stay in sync.
"""
import argparse
import os
import subprocess
import tempfile
from collections import Counter
from datetime import timedelta
from pathlib import Path

from faster_whisper import WhisperModel  # pip install faster-whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment

# ---------------------------------------------------------------------------
# Speed‑up helper
# ---------------------------------------------------------------------------

def _audio_speedup(in_file: Path, factor: float) -> Path:
    """Return a temporary WAV containing `in_file` sped up (or slowed) by `factor`.

    Internally uses FFmpeg's `atempo` filter, which only supports 0.5–2.0× per
    stage; factors outside that range are achieved by chaining multiple stages.
    """
    if abs(factor - 1.0) < 1e-3:
        return in_file  # No change requested

    filters: list[str] = []
    remaining = factor
    while not 0.5 <= remaining <= 2.0:
        hop = 2.0 if remaining > 2.0 else 0.5
        filters.append(f"atempo={hop}")
        remaining /= hop
    filters.append(f"atempo={remaining}")
    atempo_chain = ",".join(filters)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(in_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-filter:a",
        atempo_chain,
        tmp.name,
    ]
    subprocess.run(cmd, check=True)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Diarization helpers
# ---------------------------------------------------------------------------

def diarize(audio_file: Path, hf_token: str | None = None):
    """Run pyannote's pretrained diarization pipeline once per file."""
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=hf_token
    )
    return pipe(str(audio_file))


def add_speaker_labels(segments, annotation):
    """Attach the most‑dominant speaker label to each Whisper segment."""
    labelled = []
    for seg in segments:
        window = Segment(seg.start, seg.end)
        overlaps = annotation.crop(window, mode="intersection")
        if not overlaps:
            speaker = "UNK"
        else:
            tally = Counter()
            for turn, _, spk in overlaps.itertracks(yield_label=True):
                ov_start = max(seg.start, turn.start)
                ov_end = min(seg.end, turn.end)
                tally[spk] += ov_end - ov_start
            speaker, _ = tally.most_common(1)[0]
        seg.speaker = speaker
        labelled.append(seg)
    return labelled


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------
def _fmt_ts(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    millis = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def write_srt(segments, out_path: Path):
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, seg in enumerate(segments, start=1):
            fh.write(f"{idx}\n")
            fh.write(f"{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}\n")
            fh.write(f"[{seg.speaker}] {seg.text.strip()}\n\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Transcribe with faster‑whisper + diarization (+ optional speed hack)"
    )
    ap.add_argument("audio", type=Path, help="Input WAV/MP3/FLAC/OGG file")
    ap.add_argument("-m", "--model", default="largev3-int8", help="CTranslate2 model name or path")
    ap.add_argument("-o", "--output_dir", type=Path, default="results", help="Folder to save transcripts")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--beam_size", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0, help="Audio speed‑up factor (e.g. 1.5 or 2.0)")

    args = ap.parse_args()

    # 1. Optional speed‑up preprocessing (for ASR **only**)
    audio_for_asr = _audio_speedup(args.audio, args.speed)

    # 2. Whisper ASR
    model = WhisperModel(args.model, device=args.device, compute_type="int8_float16")
    print("→ transcribing …")
    seg_gen, info = model.transcribe(
        str(audio_for_asr), beam_size=args.beam_size, word_timestamps=True
    )
    segments = list(seg_gen)  # materialise generator so we can iterate multiple times
    print(f"Detected language: {info.language}; duration {info.duration:.1f}s (processed)")

    # 3. Rescale timestamps back to original timeline if speed ≠ 1
    if abs(args.speed - 1.0) > 1e-3:
        for seg in segments:
            seg.start *= args.speed
            seg.end *= args.speed

    # 4. Diarization on **original** audio
    print("→ diarizing …")
    diar = diarize(args.audio, os.getenv("HUGGINGFACE_TOKEN"))

    # 5. Merge speakers with ASR segments
    segments = add_speaker_labels(segments, diar)

    # 6. Write SRT
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.audio.stem}.srt"
    write_srt(segments, out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
