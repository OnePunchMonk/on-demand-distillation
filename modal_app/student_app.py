"""Always-on student endpoint. Small model, kept warm (min_containers=1).

Deploy: modal deploy modal_app/student_app.py
"""
import modal

from modal_app.common import build_llm, generate_text, vllm_image

app = modal.App("on-demand-distill-student")

STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@app.cls(
    image=vllm_image,
    gpu="A10G",
    min_containers=1,          # stay warm: this is the always-on path
    scaledown_window=600,
)
class Student:
    @modal.enter()
    def load(self):
        self.llm = build_llm(STUDENT_MODEL)

    @modal.method()
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        return generate_text(self.llm, prompt, temperature, add_confidence=True)


@app.function(image=vllm_image)
def generate(prompt: str, temperature: float = 0.2) -> str:
    return Student().generate.remote(prompt, temperature)
