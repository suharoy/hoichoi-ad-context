"""Frozen V1 Bengali ASR backend."""

from pathlib import Path

import onnx_asr


class IndicConformerASR:
    def __init__(self) -> None:
        self.model = onnx_asr.load_model(
            "OpenVoiceOS/ai4bharat-indicconformer-bn-onnx",
            quantization="int8",
            providers=["CPUExecutionProvider"],
        )
        print(
            "ASR initialized: backend=onnx-asr, "
            "model=OpenVoiceOS/ai4bharat-indicconformer-bn-onnx, "
            "quantization=int8, device=cpu",
            flush=True,
        )

    def recognize(self, audio_path: str | Path) -> str:
        return self.model.recognize(str(audio_path)).strip()
