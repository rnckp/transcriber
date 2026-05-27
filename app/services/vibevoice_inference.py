"""Local VibeVoice ASR inference adapter.

Adapted from Microsoft VibeVoice's MIT-licensed
``demo/vibevoice_asr_gradio_demo.py``.
"""

import logging
import time
from typing import Any

from vibevoice.modular.modeling_vibevoice_asr import (
    VibeVoiceASRForConditionalGeneration,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

logger = logging.getLogger(__name__)


class VibeVoiceASRInference:
    """Simple inference wrapper for VibeVoice ASR models."""

    def __init__(
        self,
        model_path: str,
        device: str,
        dtype: Any,
        attn_implementation: str,
    ) -> None:
        logger.info("Loading VibeVoice ASR model from %s", model_path)
        self.processor = VibeVoiceASRProcessor.from_pretrained(model_path)

        logger.info("Using VibeVoice attention implementation: %s", attn_implementation)
        if device == "mps":
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=None,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
            self.model = self.model.to("mps")
        elif device == "auto":
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                model_path,
                dtype=dtype,
                device_map="auto",
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        else:
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=device,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
            self.model = self.model.to(device)

        self.device = (
            device if device != "auto" else next(self.model.parameters()).device
        )
        self.model.eval()

        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        logger.info(
            "Loaded VibeVoice ASR model on %s with %.2fB parameters",
            self.device,
            total_params / 1_000_000_000,
        )

    def transcribe(
        self,
        audio_path: str,
        sample_rate: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        context_info: str | None = None,
        streamer: Any | None = None,
    ) -> dict[str, Any]:
        """Transcribe one audio file with a loaded VibeVoice ASR model."""
        import torch

        inputs = self.processor(
            audio=audio_path,
            sampling_rate=sample_rate,
            return_tensors="pt",
            add_generation_prompt=True,
            context_info=context_info,
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else None,
            "top_p": top_p if do_sample else None,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        generation_config = {
            key: value for key, value in generation_config.items() if value is not None
        }

        start_time = time.time()
        input_ids = inputs["input_ids"][0]
        input_tokens = self._count_input_tokens(input_ids)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_config)

        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        generated_text = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        try:
            transcription_segments = self.processor.post_process_transcription(
                generated_text,
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Failed to parse VibeVoice structured output", exc_info=True)
            transcription_segments = []

        return {
            "raw_text": generated_text,
            "segments": transcription_segments,
            "generation_time": time.time() - start_time,
            "input_tokens": input_tokens,
        }

    def _count_input_tokens(self, input_ids: Any) -> dict[str, int]:
        pad_id = self.processor.pad_id
        padding_mask = input_ids == pad_id
        num_padding_tokens = int(padding_mask.sum().item())

        speech_start_id = self.processor.speech_start_id
        speech_end_id = self.processor.speech_end_id
        num_speech_tokens = 0
        in_speech = False
        for token_id in input_ids.tolist():
            if token_id == speech_start_id:
                in_speech = True
                num_speech_tokens += 1
            elif token_id == speech_end_id:
                in_speech = False
                num_speech_tokens += 1
            elif in_speech:
                num_speech_tokens += 1

        total_input_tokens = int(input_ids.shape[0])
        return {
            "total": total_input_tokens,
            "speech": num_speech_tokens,
            "text": total_input_tokens - num_speech_tokens - num_padding_tokens,
            "padding": num_padding_tokens,
        }
