# main.py
import os
import re
import torch
import asyncio
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------
# Environment tuning (important for MIG + offload inference)
# --------------------------------------------------------------------
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "openai/gpt-oss-20b"
OFFLOAD_DIR = "offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Load tokenizer
# --------------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --------------------------------------------------------------------
# Hybrid GPU + CPU model loading (35GB compatible)
# --------------------------------------------------------------------
print("Loading model with GPU/CPU partitioning... (takes a few minutes)")

max_memory = {
    0: "34GiB",     # Your 35GB MIG slice (leave ~1GB safety margin)
    "cpu": "120GiB" # Adjust if your node RAM differs
}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    low_cpu_mem_usage=True
)

model.eval()
print("Model ready — running hybrid execution")

# --------------------------------------------------------------------
# FastAPI setup
# --------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Harmony prompt template
# --------------------------------------------------------------------
HARMONY_TEMPLATE = """<|start|>system<|message|>
{system_msg}
<|end|>

<|start|>developer<|message|>
{developer_msg}
<|end|>

<|start|>user<|message|>
{user_msg}
<|end|>

<|start|>assistant
<|channel|>final<|message|>
"""

def build_prompt(system, developer, user):
    return HARMONY_TEMPLATE.format(
        system_msg=system.strip(),
        developer_msg=developer.strip(),
        user_msg=user.strip(),
    )

FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*)", re.S)

# --------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------
def run_inference(user_prompt: str):

    system_msg = "You are a world-class Earth Observation analyst. Reasoning: high."
    developer_msg = "Always respond with EO-standard terminology. NEVER output analysis, only final results."

    prompt = build_prompt(system_msg, developer_msg, user_prompt)

    inputs = tokenizer(prompt, return_tensors="pt")

    # send tensors to first execution device
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract final channel only
    m = FINAL_RE.search(raw)
    if not m:
        cleaned = re.sub(r"<\|.*?\|>", "", raw).strip()
        return JSONResponse({"response": cleaned})

    final_text = m.group(1).strip()
    final_text = re.sub(r"<\|channel\|>analysis.*", "", final_text, flags=re.S).strip()

    return JSONResponse({"response": final_text})

# --------------------------------------------------------------------
# Async endpoint
# --------------------------------------------------------------------
@app.post("/ask_gptoss")
async def ask_gptoss(prompt: str = Form(...)):
    return await asyncio.to_thread(run_inference, prompt)

# --------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "GPT-OSS hybrid GPU/CPU server running"}
