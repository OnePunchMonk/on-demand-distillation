"""On-demand teacher endpoint. Larger model, scaled to zero — only pay for it
on escalation.

Deploy: modal deploy modal_app/teacher_app.py
"""
import modal

from modal_app.common import build_llm, generate_text, vllm_image

app = modal.App("on-demand-distill-teacher")

TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct"


@app.cls(
    image=vllm_image,
    gpu="A100-80GB:4",
    min_containers=0,          # scale to zero: only invoked on escalation
    scaledown_window=120,
)
class Teacher:
    @modal.enter()
    def load(self):
        self.llm = build_llm(TEACHER_MODEL)

    @modal.method()
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        return generate_text(self.llm, prompt, temperature, add_confidence=False)


@app.function(image=vllm_image)
def generate(prompt: str, temperature: float = 0.2) -> str:
    return Teacher().generate.remote(prompt, temperature)
