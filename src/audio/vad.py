from __future__ import annotations

import collections
import contextlib
import wave
from dataclasses import dataclass
from pathlib import Path

import webrtcvad


@dataclass(frozen=True)
class Frame:
    bytes: bytes
    timestamp: float
    duration: float


def read_wave(path: str | Path) -> tuple[bytes, int]:
    path = Path(path)

    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()

        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels.")

        if sample_width != 2:
            raise ValueError(
                f"Expected 16-bit PCM WAV, got sample width={sample_width}."
            )

        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(
                f"Unsupported WebRTC VAD sample rate: {sample_rate}"
            )

        pcm = wf.readframes(wf.getnframes())

    return pcm, sample_rate


def generate_frames(
    frame_duration_ms: int,
    audio: bytes,
    sample_rate: int,
):
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    bytes_per_frame = samples_per_frame * 2

    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0

    while offset + bytes_per_frame <= len(audio):
        yield Frame(
            bytes=audio[offset : offset + bytes_per_frame],
            timestamp=timestamp,
            duration=duration,
        )

        timestamp += duration
        offset += bytes_per_frame


def collect_speech_segments(
    sample_rate: int,
    frame_duration_ms: int,
    padding_duration_ms: int,
    vad: webrtcvad.Vad,
    frames,
) -> list[dict[str, float]]:
    num_padding_frames = padding_duration_ms // frame_duration_ms

    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False
    segment_start: float | None = None

    segments: list[dict[str, float]] = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))

            voiced = sum(1 for _, speech in ring_buffer if speech)

            if (
                len(ring_buffer) == ring_buffer.maxlen
                and voiced > 0.8 * ring_buffer.maxlen
            ):
                triggered = True
                segment_start = ring_buffer[0][0].timestamp
                ring_buffer.clear()

        else:
            ring_buffer.append((frame, is_speech))

            unvoiced = sum(
                1 for _, speech in ring_buffer if not speech
            )

            if (
                len(ring_buffer) == ring_buffer.maxlen
                and unvoiced > 0.8 * ring_buffer.maxlen
            ):
                segment_end = (
                    frame.timestamp + frame.duration
                )

                assert segment_start is not None

                segments.append(
                    {
                        "start": round(segment_start, 3),
                        "end": round(segment_end, 3),
                    }
                )

                triggered = False
                segment_start = None
                ring_buffer.clear()

    if triggered and segment_start is not None:
        final_end = frame.timestamp + frame.duration
        segments.append(
            {
                "start": round(segment_start, 3),
                "end": round(final_end, 3),
            }
        )

    return segments


def detect_speech(
    audio_path: str | Path,
    *,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
    padding_duration_ms: int = 300,
) -> list[dict[str, float]]:
    pcm, sample_rate = read_wave(audio_path)

    vad = webrtcvad.Vad(aggressiveness)

    frames = generate_frames(
        frame_duration_ms,
        pcm,
        sample_rate,
    )

    return collect_speech_segments(
        sample_rate,
        frame_duration_ms,
        padding_duration_ms,
        vad,
        frames,
    )