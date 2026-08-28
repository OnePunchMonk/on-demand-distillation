"""Always-on student endpoint. Small model, kept warm (min_containers=1).

Reads /data/serving/current_checkpoint.txt on cold start: if a LoRA adapter
has been promoted by a post-training cycle, it's applied on top of the base
model; otherwise it serves the plain base model. This is the point where
the post-training loop actually changes what's served — see
training/promote.py.

Deploy: modal deploy modal_app/student_app.py
"""
import modal

from modal_app.common import build_llm, generate_text, vllm_image

app = modal.App("on-demand-distill-student")

STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
volume = modal.Volume.from_name("on-demand-distill-data", create_if_missing=True)


def resolve_adapter_path() -> str | None:
    from pathlib import Path

    pointer = Path("/data/serving/current_checkpoint.txt")
    if not pointer.exists():
        return None
    value = pointer.read_text().strip()
    return None if value in ("", "base") else value


@app.cls(
    image=vllm_image,
    gpu="A10G",
    min_containers=1,          # stay warm: this is the always-on path
    scaledown_window=600,
    volumes={"/data": volume},
)
class Student:
    @modal.enter()
    def load(self):
        volume.reload()
        adapter_path = resolve_adapter_path()
        self.llm = build_llm(STUDENT_MODEL, lora_adapter_path=adapter_path)
        self.adapter_path = adapter_path

    @modal.method()
    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        return generate_text(self.llm, prompt, temperature, add_confidence=True)

    @modal.method()
    def served_checkpoint(self) -> str:
        return self.adapter_path or "base"


@app.function(image=vllm_image, volumes={"/data": volume})
def generate(prompt: str, temperature: float = 0.2) -> str:
    return Student().generate.remote(prompt, temperature)
