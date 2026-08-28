"""Shared vLLM-on-Modal serving helper for both student and teacher apps."""
from __future__ import annotations

import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.6.3", "torch==2.4.0", "transformers>=4.45")
)

CONFIDENCE_INSTRUCTION = (
    "\n\nAfter your answer, on a new line, emit your confidence in that "
    "answer as [[confidence: 0.NN]] where 0.00 is a guess and 1.00 is certain."
)


def build_llm(model_name: str, lora_adapter_path: str | None = None):
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        enable_lora=lora_adapter_path is not None,
    )
    if lora_adapter_path is not None:
        llm._distill_lora_request = LoRARequest("student-distilled", 1, lora_adapter_path)
    else:
        llm._distill_lora_request = None
    return llm


def generate_text(llm, prompt: str, temperature: float, max_tokens: int = 1024, add_confidence: bool = False) -> str:
    from vllm import SamplingParams

    full_prompt = prompt + (CONFIDENCE_INSTRUCTION if add_confidence else "")
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    lora_request = getattr(llm, "_distill_lora_request", None)
    output = llm.generate([full_prompt], params, lora_request=lora_request)
    return output[0].outputs[0].text.strip()
