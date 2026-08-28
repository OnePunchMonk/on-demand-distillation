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


def build_llm(model_name: str):
    from vllm import LLM

    return LLM(model=model_name, trust_remote_code=True)


def generate_text(llm, prompt: str, temperature: float, max_tokens: int = 1024, add_confidence: bool = False) -> str:
    from vllm import SamplingParams

    full_prompt = prompt + (CONFIDENCE_INSTRUCTION if add_confidence else "")
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    output = llm.generate([full_prompt], params)
    return output[0].outputs[0].text.strip()
