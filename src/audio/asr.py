from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel


class ASRModel:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        print(
            f"ASR initialized: backend=faster-whisper, model={model_size}, "
            f"device={device}, compute_type={compute_type}"
        )

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> list[dict]:
        segments, info = self.model.transcribe(
            str(audio_path),
            language="bn",
            beam_size=5,
            word_timestamps=True,
            vad_filter=False,
        )

        output = []

        for segment in segments:
            words = []

            for word in segment.words or []:
                words.append(
                    {
                        "start": round(
                            float(word.start), 3
                        ),
                        "end": round(
                            float(word.end), 3
                        ),
                        "word": word.word.strip(),
                    }
                )

            output.append(
                {
                    "start": round(
                        float(segment.start), 3
                    ),
                    "end": round(
                        float(segment.end), 3
                    ),
                    "text": segment.text.strip(),
                    "words": words,
                }
            )

        return output
