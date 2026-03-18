# main.py
import os
import re
import torch
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------
# Disable FlashAttention (important for MIG stability)
# --------------------------------------------------------------------
os.environ["DISABLE_FLASH_ATTENTION"] = "1"

# --------------------------------------------------------------------
# Load GPT-OSS-20B (MXFP4 native, NO extra quantization)
# --------------------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-20b"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,   # lighter than bf16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --------------------------------------------------------------------
# FastAPI Application
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
# Harmony Prompt Template (FORCE FINAL CHANNEL)
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

# --------------------------------------------------------------------
# Extract ONLY the final channel (ZERO analysis leakage)
# --------------------------------------------------------------------
FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*)",
    re.S
)

@app.post("/ask_gptoss")
async def ask_gptoss(prompt: str = Form(...)):

    system_msg = "You are a world-class Earth Observation analyst. Reasoning: high."
    developer_msg = "Always respond with EO-standard terminology. NEVER output analysis, only final results."
    
    harmony_prompt = build_prompt(system_msg, developer_msg, prompt)

    # Tokenize with strict truncation (CRITICAL for 35GB)
    inputs = tokenizer(
        harmony_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768   # SAFE limit
    ).to("cuda")

    # Generate with tight limits (prevents KV explosion)
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,   # SAFE limit
        do_sample=False,
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract <|channel|>final
    m = FINAL_RE.search(raw)
    if not m:
        cleaned = re.sub(r"<\|.*?\|>", "", raw).strip()
        return JSONResponse({"response": cleaned})

    final_text = m.group(1).strip()

    # Remove any rogue analysis
    final_text = re.sub(r"<\|channel\|>analysis.*", "", final_text, flags=re.S).strip()

    return JSONResponse({"response": final_text})

# --------------------------------------------------------------------
# Health Endpoint
# --------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "GPT-OSS35 Final-Only Server Running"}